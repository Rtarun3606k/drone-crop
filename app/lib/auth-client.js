"use client";

import {
  signIn as nextAuthSignIn,
  signOut as nextAuthSignOut,
  useSession,
} from "next-auth/react";

export function signIn(provider, options) {
  return nextAuthSignIn(provider, options);
}

export function signOut(options) {
  return nextAuthSignOut(options);
}

// Custom hook to check if the user has a specific role
export function useRole(requiredRole) {
  const { data: session, status } = useSession();
  const isLoading = status === "loading";
  const hasRole = session?.user?.role === requiredRole;

  return {
    hasRole,
    isLoading,
    session,
    status,
  };
}

// Custom hook to check if the user is an admin
export function useIsAdmin() {
  return useRole("ADMIN");
}

/**
 * Helper function to check if a user session has a specific role
 * @param {Object} session - The user session
 * @param {string} role - The required role
 * @return {boolean} - True if the user has the required role
 */
export function checkRoleAccess(session, role) {
  if (!session || !session.user) return false;
  return session.user.role === role;
}

export function useAdmin() {
  return useRole("ADMIN");
}
