"use client";

import React, { useEffect } from "react";
import { useSession } from "next-auth/react";
import { useRouter } from "next/navigation";

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
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
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
      <div className="bg-white dark:bg-gray-900 shadow-lg rounded-lg p-8 w-full max-w-md border border-gray-200 dark:border-gray-800">
        <div className="flex flex-col items-center">
          {user.image ? (
            <img
              src={user.image}
              alt={user.name || "User avatar"}
              className="w-24 h-24 rounded-full border-4 border-green-500 mb-4 object-cover"
            />
          ) : (
            <div className="w-24 h-24 rounded-full border-4 border-green-500 mb-4 bg-green-100 dark:bg-green-800 flex items-center justify-center">
              <span className="text-green-700 dark:text-green-300 font-bold text-2xl">
                {user.name?.charAt(0) || user.email?.charAt(0) || "U"}
              </span>
            </div>
          )}
          <h2 className="text-2xl font-bold mb-1 text-gray-900 dark:text-white">
            {user.name}
          </h2>
          <p className="text-gray-500 dark:text-gray-400 mb-4">{user.email}</p>

          <div className="w-full mt-6 space-y-4">
            <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg">
              <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400">
                Account Information
              </h3>
              <div className="mt-2 grid grid-cols-2 gap-4">
                <div>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    Account ID
                  </p>
                  <p className="text-sm font-medium text-gray-900 dark:text-white">
                    {user.id?.substring(0, 8) || "N/A"}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    Member Since
                  </p>
                  <p className="text-sm font-medium text-gray-900 dark:text-white">
                    {new Date().toLocaleDateString()}
                  </p>
                </div>
              </div>
            </div>

            <button
              className="w-full py-3 px-4 bg-green-600 hover:bg-green-700 text-white font-medium rounded-lg transition-colors"
              onClick={() => router.push("/dashboard")}
            >
              Go to Dashboard
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
