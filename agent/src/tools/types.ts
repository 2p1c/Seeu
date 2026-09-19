export interface Tool {
  name: string;
  description: string;
  parameters: {
    type: "object";
    properties: Record<string, { type: string; description?: string }>;
    required?: readonly string[];
  };
  execute: (args: Record<string, unknown>) => Promise<string>;
}
