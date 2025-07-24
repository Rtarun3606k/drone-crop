// ./lib/swagger.js

import { createSwaggerSpec } from "next-swagger-doc";

export const getApiDocs = async () => {
  const spec = createSwaggerSpec({
    apiFolder: "app/api", // Folder containing your Next.js API routes
    definition: {
      openapi: "3.0.0",
      info: {
        title: "Next Swagger API Example",
        version: "1.0.0",
        description: "API documentation for the Next.js App Router project.",
        contact: {
          name: "Tarun Nayaka Sobhi",
          email: "you@example.com", // Replace with your real contact if needed
        },
      },
      servers: [
        {
          url:
            process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:3000/api",
          description: "Development server",
        },
      ],
      components: {
        securitySchemes: {
          SessionCookieAuth: {
            type: "apiKey",
            in: "cookie",
            name: "next-auth.session-token",
          },
        },
        schemas: {
          // === Enums ===
          Language: {
            type: "string",
            enum: ["En", "Ta", "Hi", "Te", "Ml", "Kn"],
          },
          Role: {
            type: "string",
            enum: ["USER", "ADMIN"],
          },

          // === User ===
          User: {
            type: "object",
            properties: {
              id: { type: "string" },
              name: { type: "string", nullable: true },
              email: { type: "string", nullable: true },
              emailVerified: {
                type: "string",
                format: "date-time",
                nullable: true,
              },
              image: { type: "string", nullable: true },
              role: { $ref: "#/components/schemas/Role" },
              defaultLanguage: { $ref: "#/components/schemas/Language" },
              timezone: { type: "string", example: "UTC" },
              metadata: { type: "object", nullable: true },
            },
          },

          // === Batch ===
          Batch: {
            type: "object",
            properties: {
              id: { type: "string" },
              name: { type: "string" },
              cropType: { type: "string" },
              imagesZipURL: { type: "string" },
              preferredLanguage: { $ref: "#/components/schemas/Language" },
              isModelCompleted: { type: "boolean" },
              isDescCompleted: { type: "boolean" },
              isAudioCompleted: { type: "boolean" },
              hasExecutionFailed: { type: "boolean" },
              imagesCount: { type: "integer" },
              sessionId: { type: "string" },
              description: { type: "string", nullable: true },
              audioURL: { type: "string", nullable: true },
              pdfURL: { type: "string", nullable: true },
              createdAt: { type: "string", format: "date-time" },
              updatedAt: { type: "string", format: "date-time" },
              metadata: { type: "object", nullable: true },

              descriptions: {
                type: "array",
                items: { $ref: "#/components/schemas/Description" },
              },
            },
          },

          // === Description ===
          Description: {
            type: "object",
            properties: {
              id: { type: "string" },
              batchId: { type: "string" },
              language: { $ref: "#/components/schemas/Language" },
              longDescription: { type: "string" },
              shortDescription: { type: "string" },
              wordCount: { type: "integer", nullable: true },
              confidence: { type: "number", format: "float", nullable: true },
              createdAt: { type: "string", format: "date-time" },
              updatedAt: { type: "string", format: "date-time" },
            },
          },
        },
      },

      security: [
        {
          SessionCookieAuth: [],
        },
      ],
    },
  });

  return spec;
};
