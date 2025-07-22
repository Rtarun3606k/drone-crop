import NextAuth from "next-auth";
import Google from "next-auth/providers/google";
import { PrismaAdapter } from "@auth/prisma-adapter";
import { prisma } from "./lib/prisma-server";

const usingPrismaAdapter = !!prisma && typeof prisma.user !== "undefined";

export const { handlers, auth, signIn, signOut } = NextAuth({
  adapter: usingPrismaAdapter ? PrismaAdapter(prisma) : undefined,
  debug: false,
  trustHost: true,
  providers: [
    Google({
      clientId: process.env.AUTH_GOOGLE_ID,
      clientSecret: process.env.AUTH_GOOGLE_SECRET,
    }),
  ],
  secret: process.env.AUTH_SECRET,
  session: {
    strategy: "database",
  },
  callbacks: {
    async session({ session, user }) {
      if (session.user) {
        session.user.id = user.id;
        session.user.role = user.role; // Assuming 'role' is on your user model
      }
      return session;
    },
    // The signIn callback is no longer needed for this logic.
    // You can keep it if you have other authorization checks.
    async signIn() {
      return true;
    },
  },
  // Add the events callback here
  events: {
    async createUser({ user }) {
      // This event fires right after a new user is created in the database.
      // 'user.id' here is the final, correct MongoDB ObjectId.
      if (user.id && user.email) {
        const emailPrefix = user.email.slice(0, 4);
        const idSuffix = user.id.slice(-4);
        const mobileId = emailPrefix + idSuffix;

        await prisma.user.update({
          where: { id: user.id },
          data: { mobileId: mobileId },
        });
      }
    },
  },
  pages: {
    signIn: "/login",
    error: "/login",
  },
});
