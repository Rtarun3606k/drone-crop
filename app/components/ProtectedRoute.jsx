"use client";

import { useSession } from "next-auth/react";
import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { checkRoleAccess } from "@/app/lib/auth-client";

/**
 * A component that protects routes based on authentication and role requirements
 *
 * @param {Object} props - Component props
 * @param {React.ReactNode} props.children - The protected content
 * @param {string} [props.requiredRole] - Optional role requirement (e.g., "ADMIN")
 * @param {string} [props.redirectTo="/login"] - Where to redirect if access is denied
 * @param {React.ReactNode} [props.loadingComponent] - Optional custom loading component
 */
export default function ProtectedRoute({
  children,
  requiredRole,
  redirectTo = "/login",
  loadingComponent,
}) {
  const { data: session, status } = useSession();
  const router = useRouter();
  const isLoading = status === "loading";

  // Check if user is authenticated and has the required role (if specified)
  const hasAccess =
    status === "authenticated" &&
    (!requiredRole || checkRoleAccess(session, requiredRole));

  useEffect(() => {
    // If authentication is complete (not loading) and user doesn't have access, redirect
    if (!isLoading && !hasAccess) {
      router.push(redirectTo);
    }
  }, [isLoading, hasAccess, router, redirectTo]);

  // Show loading state
  if (isLoading) {
    return (
      loadingComponent || (
        <div className="min-h-screen flex items-center justify-center">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-green-500"></div>
        </div>
      )
    );
  }

  // Don't render anything if not authenticated (will redirect via useEffect)
  if (!hasAccess) return null;

  // Render protected content if user has access
  return <>{children}</>;
}
