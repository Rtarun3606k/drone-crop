"use client";

import { SessionProvider } from "next-auth/react";

export default function NextAuthProvider({ children }) {
  return (
    <SessionProvider refetchInterval={5 * 60} refetchOnWindowFocus={true}>
      {children}
    </SessionProvider>
  );
}
