"use client";

import React, { useEffect } from "react";
import { useSession } from "next-auth/react";
import { useRouter } from "next/navigation";
import SessionManager from "@/app/components/SessionManager";

export default function ProfilePage() {
  const { data: session, status } = useSession();
  const router = useRouter();

  useEffect(() => {
    if (status === "unauthenticated") {
      router.push("/login");
    }
  }, [status, router]);

  // Show loading state while checking authentication
  if (status === "loading") {
    return (
      <div className="min-h-screen bg-transparent flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-green-500"></div>
      </div>
    );
  }

  // Don't render anything if not authenticated (will redirect via useEffect)
  if (!session) {
    return null;
  }

  const user = session.user;

  return (
    <div className="min-h-screen bg-transparent flex flex-col items-center py-24">
      <div className="w-full max-w-3xl px-4">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-8">
          Your Profile
        </h1>

        <div className="bg-white dark:bg-gray-900 shadow-lg rounded-lg p-8 mb-8 border border-gray-200 dark:border-gray-800">
          <div className="flex flex-col md:flex-row md:items-center">
            <div className="flex-shrink-0 flex justify-center md:justify-start mb-6 md:mb-0 md:mr-6">
              {user.image ? (
                <img
                  src={user.image}
                  alt={user.name || "User avatar"}
                  className="w-24 h-24 rounded-full border-4 border-green-500 object-cover"
                />
              ) : (
                <div className="w-24 h-24 rounded-full border-4 border-green-500 bg-green-100 dark:bg-green-800 flex items-center justify-center">
                  <span className="text-green-700 dark:text-green-300 font-bold text-2xl">
                    {user.name?.charAt(0) || user.email?.charAt(0) || "U"}
                  </span>
                </div>
              )}
            </div>

            <div className="flex-grow">
              <h2 className="text-2xl font-bold text-center md:text-left text-gray-900 dark:text-white">
                {user.name}
              </h2>
              <p className="text-gray-500 dark:text-gray-400 text-center md:text-left mb-4">
                {user.email}
              </p>

              <div className="flex flex-wrap gap-2 justify-center md:justify-start">
                <span className="px-3 py-1 bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200 text-sm rounded-full">
                  Google Account
                </span>
                {user.role && (
                  <span
                    className={`px-3 py-1 text-sm rounded-full ${
                      user.role === "ADMIN"
                        ? "bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200"
                        : "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200"
                    }`}
                  >
                    {user.role} Role
                  </span>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Session Manager Component */}
        <SessionManager />
      </div>
    </div>
  );
}
