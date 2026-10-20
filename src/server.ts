import { createWorkersAI } from "workers-ai-provider";
import { callable, routeAgentRequest, type Schedule } from "agents";
import { scheduleSchema } from "agents/schedule";
import { AIChatAgent, type OnChatMessageOptions } from "@cloudflare/ai-chat";
import { convertToModelMessages, pruneMessages, stepCountIs, streamText, tool } from "ai";
import { z } from "zod";

export type Result = { topic: string; score: number; at: string };
export type CoachState = { results: Result[] };

export interface Env {
  AI: any;
}

export class ChatAgent extends AIChatAgent<Env, CoachState> {
  initialState: CoachState = { results: [] };
  maxPersistedMessages = 100;
  chatRecovery = true;
  waitForMcpConnections = true;

  onStart() {
    this.mcp.configureOAuthCallback({
      customHandler: (result) => {
        if (result.authSuccess) {
          return new Response("<script>window.close();</script>", {
            headers: { "content-type": "text/html" },
            status: 200
          });
        }
        return new Response(
          `Authentication Failed: ${result.authError || "Unknown error"}`,
          { headers: { "content-type": "text/plain" }, status: 400 }
        );
      }
    });
  }

  @callable()
  async addServer(name: string, url: string) {
    return await this.addMcpServer(name, url);
  }

  @callable()
  async removeServer(serverId: string) {
    await this.removeMcpServer(serverId);
  }

  async onChatMessage(_onFinish: unknown, options?: OnChatMessageOptions) {
    const mcpTools = this.mcp.getAITools();
    const workersai = createWorkersAI({ binding: this.env.AI });

    // Clean historical messages array to drop background tool retry chains instantly
    const rawMessages = await convertToModelMessages(this.messages);
    const safeMessages = rawMessages.slice(-2);

    // Identify user scheduling intent from recent turns
    const userIntentText = safeMessages.map(m => typeof m.content === 'string' ? m.content : '').join(' ').toLowerCase();
    const shouldExposeScheduleTool = userIntentText.includes("schedule") || userIntentText.includes("remind");

    // Construct the explicit tools map
    const baseTools: Record<string, any> = {
      ...mcpTools,

      recordResult: tool({
        description: "Save the score for an interview topic after grading",
        inputSchema: z.object({
          topic: z.string().describe("short topic, e.g. database indexing"),
          score: z.number().describe("Provide a raw integer score from 0 to 10")
        }),
        execute: async ({ topic, score }) => {
          const currentResults = this.state?.results || [];
          const updatedResults = [
            ...currentResults, 
            { topic, score: Math.round(score), at: new Date().toISOString() }
          ];

          this.setState({
            results: updatedResults
          });
          
          await this.save(); 
          return `Saved ${topic}: ${score}/10`;
        }
      }),

      getProgress: tool({
        description: "Get saved scores grouped by topic to find weak areas",
        inputSchema: z.object({}),
        execute: async () => {
          const currentResults = this.state?.results || [];
          const by: Record<string, number[]> = {};
          for (const r of currentResults) {
            if (r && r.topic) (by[r.topic] ??= []).push(r.score);
          }
          return Object.entries(by).map(([topic, s]) => ({
            topic,
            attempts: s.length,
            average: Math.round((s.reduce((a, b) => a + b, 0) / s.length) * 10) / 10
          }));
        }
      }),

      calculate: tool({
        description: "Perform a math calculation with two numbers. Requires user approval for large numbers.",
        inputSchema: z.object({
          a: z.number().describe("First number"),
          b: z.number().describe("Second number"),
          operator: z.enum(["+", "-", "*", "/", "%"]).describe("Arithmetic operator")
        }),
        needsApproval: async ({ a, b }) => Math.abs(a) > 1000 || Math.abs(b) > 1000,
        execute: async ({ a, b, operator }) => {
          const ops: Record<string, (x: number, y: number) => number> = {
            "+": (x, y) => x + y,
            "-": (x, y) => x - y,
            "*": (x, y) => x * y,
            "/": (x, y) => x / y,
            "%": (x, y) => x % y
          };
          if (operator === "/" && b === 0) {
            return { error: "Division by zero" };
          }
          return {
            expression: `${a} ${operator} ${b}`,
            result: ops[operator](a, b)
          };
        }
      })
    };

    // Dynamically mount scheduleTask ONLY if user explicitly asked for it
    if (shouldExposeScheduleTool) {
      baseTools.scheduleTask = tool({
        description: "Schedule an interview reminder to be executed at a later time.",
        inputSchema: scheduleSchema,
        execute: async ({ when, description }) => {
          if (when.type === "no-schedule") return "Not a valid schedule input";
          const input = when.type === "scheduled" ? when.date : when.type === "delayed" ? when.delayInSeconds : when.type === "cron" ? when.cron : null;
          if (!input) return "Invalid schedule type";
          try {
            this.schedule(input, "executeTask", description, { idempotent: true });
            return `Task scheduled: "${description}" (${when.type}: ${input})`;
          } catch (error) {
            return `Error scheduling task: ${error}`;
          }
        }
      });
    }

    const result = streamText({
      model: workersai("@cf/meta/llama-3.1-8b-instruct-fp8"),
      maxSteps: 5, // 👈 CRITICAL: Allows multi-step tool calls + text generation in a single turn
      system: `You are a friendly but rigorous interview coach for Python backend and AI/ML roles. 
Strict Rules:
1. Ask ONE technical question at a time. Do NOT talk about scheduling or dates.
2. After the user answers your question, evaluate their response. 
3. FIRST, immediately call the 'recordResult' tool to save their score. Do NOT wrap this in markdown or text. Let the system run the tool step.
4. SECOND, after the tool completes, output a short, encouraging text response to the user providing a score out of 10 and a brief breakdown of what was good and what was missing.
5. NEVER mention dates, calendars, or timelines unless explicitly asked.
Keep responses concise, natural, and friendly. Do not output raw JSON blocks.`,
      messages: pruneMessages({
        messages: safeMessages,
        toolCalls: "before-last-2-messages",
        reasoning: "before-last-message"
      }),
      tools: baseTools,
      stopWhen: stepCountIs(20),
      abortSignal: options?.abortSignal
    });

    return result.toUIMessageStreamResponse();
  }

  async executeTask(description: string, _task: Schedule<string>) {
    console.log(`Executing scheduled task: ${description}`);
    this.broadcast(
      JSON.stringify({
        type: "scheduled-task",
        description,
        timestamp: new Date().toISOString()
      })
    );
  }
}

export default {
  async fetch(request: Request, env: Env) {
    return (
      (await routeAgentRequest(request, env)) ||
      new Response("Not found", { status: 404 })
    );
  }
} satisfies ExportedHandler<Env>;
