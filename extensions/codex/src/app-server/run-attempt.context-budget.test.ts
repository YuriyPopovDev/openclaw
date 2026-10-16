import fs from "node:fs/promises";
import path from "node:path";
import type { HarnessContextEngine as ContextEngine } from "openclaw/plugin-sdk/agent-harness-runtime";
import { openFileBackedSessionManagerForTest } from "openclaw/plugin-sdk/agent-runtime-test-contracts";
import { describe, expect, it, vi } from "vitest";
import {
  assistantMessage,
  setupRunAttemptTestHooks,
  tempDir,
  userMessage,
} from "./run-attempt-test-harness.js";
import {
  createContextEngine,
  createParams,
  createStartedThreadHarness,
  getRequestInputText,
  makeThreadBootstrapBinding,
  runCodexAppServerAttempt,
  writeCodexAppServerBinding,
} from "./run-attempt.context-engine.test-support.js";

setupRunAttemptTestHooks();

describe("runCodexAppServerAttempt native context budget", () => {
  it.each([
    ["matching model", "gpt-5.4-codex", false, undefined, undefined, 238_400, false],
    [
      "provider-qualified model",
      "openai/gpt-5.4-codex",
      false,
      undefined,
      undefined,
      238_400,
      false,
    ],
    ["different model", "gpt-5.4-codex-mini", false, undefined, undefined, undefined, false],
    ["newer unreported turn", "gpt-5.4-codex", true, undefined, undefined, undefined, false],
    ["lowered authored cap", "gpt-5.4-codex", false, 32_000, undefined, 12_000, true],
    ["resolved budget", "gpt-5.4-codex", false, undefined, 80_000, 60_000, false],
    ["authored cap before observation", undefined, false, 500_000, undefined, 250_000, false],
    ["unknown window", undefined, false, undefined, undefined, undefined, false],
  ] as const)(
    "supplies the assembly budget for %s",
    async (
      _case,
      bindingModel,
      newerUnreportedTurn,
      authoredCap,
      resolvedBudget,
      expectedBudget,
      reservesInput,
    ) => {
      const sessionFile = path.join(tempDir, "session.jsonl");
      const workspaceDir = path.join(tempDir, "workspace");
      const agentDir = path.join(tempDir, "agent");
      if (bindingModel) {
        await writeCodexAppServerBinding(sessionFile, {
          ...makeThreadBootstrapBinding({
            threadId: "thread-reported-window",
            cwd: workspaceDir,
            policyFingerprint:
              '{"schemaVersion":1,"engineId":"lossless-claw","ownsCompaction":true,"projectionMaxChars":24000}',
            epoch: "epoch-1",
          }),
          model: bindingModel,
          modelProvider: "openai",
        });
        const rolloutDir = path.join(agentDir, "codex-home", "sessions");
        await fs.mkdir(rolloutDir, { recursive: true });
        await fs.writeFile(
          path.join(rolloutDir, "rollout-thread-reported-window.jsonl"),
          [
            { type: "turn_context", payload: { model: "gpt-5.4-codex" } },
            {
              type: "event_msg",
              payload: {
                type: "token_count",
                info: { last_token_usage: { total_tokens: 20_000 }, model_context_window: 258_400 },
              },
            },
            ...(newerUnreportedTurn
              ? [{ type: "turn_context", payload: { model: "gpt-5.4-codex" } }]
              : []),
          ]
            .map((line) => `${JSON.stringify(line)}\n`)
            .join(""),
        );
      }
      const assemble = vi.fn<ContextEngine["assemble"]>(async () => ({
        messages: [],
        estimatedTokens: 0,
      }));
      const contextEngine = createContextEngine({ assemble });
      const params = createParams(sessionFile, workspaceDir);
      params.agentDir = agentDir;
      params.contextEngine = contextEngine;
      params.contextTokenBudget = resolvedBudget;
      params.authoredContextTokenCap = authoredCap;
      const harness = createStartedThreadHarness();

      const run = runCodexAppServerAttempt(params);
      await harness.waitForMethod("turn/start");

      expect(assemble).toHaveBeenCalledOnce();
      const assembly = assemble.mock.calls[0]?.[0];
      expect(assembly?.runtimeSettings?.limits.promptTokenBudget).toBe(
        assembly?.tokenBudget ?? null,
      );
      if (reservesInput && expectedBudget !== undefined) {
        expect(assembly?.tokenBudget).toBeGreaterThan(0);
        expect(assembly?.tokenBudget).toBeLessThan(expectedBudget);
      } else {
        expect(assembly?.tokenBudget).toBe(expectedBudget);
      }

      await harness.completeTurn();
      await run;
    },
  );

  it("uses the runtime token budget for large Codex context-engine projections", async () => {
    const sessionFile = path.join(tempDir, "session.jsonl");
    const workspaceDir = path.join(tempDir, "workspace");
    const sessionManager = openFileBackedSessionManagerForTest(sessionFile, {
      sessionId: "session-1",
    });
    const originalObjective = "Retain our original objective: durable-budget-goal-7319.";
    const compactionSummary = "Completed investigation: durable-budget-summary-4827.";
    const retainedEntryId = sessionManager.appendMessage(userMessage(originalObjective, 1));
    sessionManager.appendCompaction(compactionSummary, retainedEntryId, 1_000);
    const longContext = `large LCM context start ${"x".repeat(30_000)} LARGE_CONTEXT_END`;
    const contextEngine = createContextEngine({
      assemble: vi.fn(async ({ messages }) => ({
        messages: [...messages, assistantMessage(longContext, 10)],
        estimatedTokens: 10_000,
        systemPromptAddition: "context-engine system",
      })),
    });
    const appServerHarness = createStartedThreadHarness();
    const attemptParams = createParams(sessionFile, workspaceDir);
    attemptParams.contextEngine = contextEngine;
    attemptParams.contextTokenBudget = 80_000;

    const attemptRun = runCodexAppServerAttempt(attemptParams);
    await appServerHarness.waitForMethod("turn/start");

    const inputText = getRequestInputText(appServerHarness);
    expect(contextEngine["assemble"]).toHaveBeenCalledWith(
      expect.objectContaining({
        tokenBudget: 60_000,
      }),
    );
    expect(contextEngine["bootstrap"]).toHaveBeenCalledWith(
      expect.objectContaining({
        runtimeContext: expect.objectContaining({ tokenBudget: 80_000 }),
      }),
    );
    expect(inputText.length).toBeGreaterThan(30_000);
    expect(inputText).toContain(originalObjective);
    expect(inputText).toContain(compactionSummary);
    expect(inputText).toContain("large LCM context start");
    expect(inputText).toContain("LARGE_CONTEXT_END");
    expect(inputText).not.toContain("[truncated ");

    await appServerHarness.completeTurn();
    await attemptRun;
  });

  it("caps assembly before projecting a large native context window", async () => {
    const sessionFile = path.join(tempDir, "session-large-window.jsonl");
    const workspaceDir = path.join(tempDir, "workspace-large-window");
    const contextEngine = createContextEngine({
      assemble: vi.fn(async () => ({
        messages: Array.from({ length: 12 }, (_, index) =>
          assistantMessage(`history-${index} ${"x".repeat(90_000)} tail-${index}`, index + 1),
        ),
        estimatedTokens: 270_000,
      })),
    });
    const harness = createStartedThreadHarness();
    const params = createParams(sessionFile, workspaceDir);
    params.contextEngine = contextEngine;
    params.contextTokenBudget = 2_000_000;
    params.prompt = "Keep the current request.";

    const run = runCodexAppServerAttempt(params);
    await harness.waitForMethod("turn/start");

    expect(contextEngine["assemble"]).toHaveBeenCalledWith(
      expect.objectContaining({ tokenBudget: 250_000 }),
    );
    const input = getRequestInputText(harness);
    const history = input.match(/<conversation_context>\n([\s\S]*)\n<\/conversation_context>/)?.[1];
    expect(history).toBeDefined();
    expect(history!.length).toBeGreaterThan(990_000);
    expect(history!.length).toBeLessThanOrEqual(1_000_000);
    expect(history).toContain("[truncated ");
    expect(history).not.toContain("history-0");
    expect(history).toContain("tail-11");
    expect(input).toContain(`Current user request:\n${params.prompt}`);
    expect(input.length).toBeLessThanOrEqual(1 << 20);

    await harness.completeTurn();
    await run;
  });
});
