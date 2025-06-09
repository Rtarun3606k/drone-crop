import { PrismaClient } from "../generated/prisma";

/**
 * PrismaClient is used server-side only.
 * This approach provides safer initialization in all environments.
 */

// Check if we're in an environment where Prisma can run
function getPrismaClient() {
  try {
    // Safety check for environment compatibility
    if (typeof window !== "undefined" || process.env.NEXT_RUNTIME === "edge") {
      return null; // Not compatible with browser or Edge runtime
    }

    // In development, we keep a single connection
    if (process.env.NODE_ENV !== "production") {
      if (!global.prisma) {
        global.prisma = new PrismaClient({
          log: ["error"],
        });
      }
      return global.prisma;
    }

    // In production, create a new instance
    return new PrismaClient({
      log: ["error"],
    });
  } catch (e) {
    console.error("Failed to initialize Prisma client:", e);
    return null;
  }
}

export const prisma = getPrismaClient();
