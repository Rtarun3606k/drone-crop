"use client";

// For client components, we should not directly instantiate PrismaClient
// Instead, use API routes to interact with the database
export const prisma = {};

// Add client-side methods that will call your API endpoints
export const getPrismaFromClient = {
  user: {
    findUnique: async (args) => {
      const res = await fetch("/api/user/profile", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(args),
      });
      return res.json();
    },
  },
  session: {
    findMany: async (args) => {
      const res = await fetch("/api/user/sessions", {
        method: "GET",
      });
      return res.json();
    },
    delete: async (args) => {
      const res = await fetch("/api/user/sessions", {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(args),
      });
      return res.json();
    },
  },
};
