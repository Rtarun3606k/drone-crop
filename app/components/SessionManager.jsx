"use client";

import { useState, useEffect } from "react";
import { useSession } from "next-auth/react";
import { formatDistanceToNow } from "date-fns";

export default function SessionManager() {
  const { data: session } = useSession();
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [deleteStatus, setDeleteStatus] = useState({
    loading: false,
    error: null,
  });

  // Fetch user sessions
  useEffect(() => {
    async function fetchSessions() {
      try {
        const response = await fetch("/api/user/sessions");
        if (!response.ok) {
          throw new Error("Failed to fetch sessions");
        }
        const data = await response.json();
        setSessions(data.sessions || []);

        // If the role from API doesn't match the session, update the session info
        if (
          data.userRole &&
          session?.user &&
          session.user.role !== data.userRole
        ) {
          console.log("Role from API:", data.userRole);
        }
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }

    if (session) {
      fetchSessions();
    }
  }, [session]);

  // Handle session termination
  const terminateSession = async (sessionToken) => {
    setDeleteStatus({ loading: true, error: null });
    try {
      const response = await fetch(
        `/api/user/sessions?sessionToken=${sessionToken}`,
        {
          method: "DELETE",
        }
      );

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Failed to terminate session");
      }

      // Update the sessions list
      setSessions(sessions.filter((s) => s.sessionToken !== sessionToken));
      setDeleteStatus({ loading: false, error: null, success: true });

      // Clear success message after 3 seconds
      setTimeout(() => {
        setDeleteStatus((prev) => ({ ...prev, success: false }));
      }, 3000);
    } catch (err) {
      setDeleteStatus({ loading: false, error: err.message });
    }
  };

  // Terminate all other sessions
  const terminateAllOtherSessions = async () => {
    setDeleteStatus({ loading: true, error: null });
    try {
      const response = await fetch(`/api/user/sessions?all=true`, {
        method: "DELETE",
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Failed to terminate sessions");
      }

      // Update the sessions list to only include current session
      setSessions(sessions.filter((s) => s.current));
      setDeleteStatus({ loading: false, error: null, success: true });

      // Clear success message after 3 seconds
      setTimeout(() => {
        setDeleteStatus((prev) => ({ ...prev, success: false }));
      }, 3000);
    } catch (err) {
      setDeleteStatus({ loading: false, error: err.message });
    }
  };

  if (loading) {
    return (
      <div className="mt-8 p-4 bg-white dark:bg-gray-800 rounded-lg shadow">
        <div className="animate-pulse flex space-x-4">
          <div className="flex-1 space-y-4 py-1">
            <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-3/4"></div>
            <div className="space-y-2">
              <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded"></div>
              <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-5/6"></div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="mt-8 p-4 bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 rounded-lg">
        Error loading sessions: {error}
      </div>
    );
  }

  return (
    <div className="mt-8">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white">
          Active Sessions
        </h3>
        {session?.user?.role && (
          <span
            className={`px-3 py-1 text-sm rounded-full ${
              session.user.role === "ADMIN"
                ? "bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200"
                : "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200"
            }`}
          >
            {session.user.role}
          </span>
        )}
      </div>

      {deleteStatus.success && (
        <div className="mb-4 p-2 bg-green-50 dark:bg-green-900/20 text-green-600 dark:text-green-400 rounded">
          Session terminated successfully
        </div>
      )}

      {deleteStatus.error && (
        <div className="mb-4 p-2 bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 rounded">
          {deleteStatus.error}
        </div>
      )}

      <div className="bg-white dark:bg-gray-800 rounded-lg shadow overflow-hidden">
        <ul className="divide-y divide-gray-200 dark:divide-gray-700">
          {sessions.length === 0 ? (
            <li className="p-4 text-gray-500 dark:text-gray-400">
              No active sessions found
            </li>
          ) : (
            sessions.map((s) => (
              <li key={s.id} className="p-4 flex items-center justify-between">
                <div>
                  <div className="flex items-center">
                    <span
                      className={`h-2 w-2 rounded-full mr-2 ${
                        s.current ? "bg-green-500" : "bg-blue-500"
                      }`}
                    ></span>
                    <span className="font-medium text-gray-900 dark:text-white">
                      {s.current ? "Current Session" : "Other Device"}
                    </span>
                    {s.current && (
                      <span className="ml-2 px-2 py-0.5 text-xs bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200 rounded-full">
                        Current
                      </span>
                    )}
                  </div>
                  <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                    Expires:{" "}
                    {formatDistanceToNow(new Date(s.expires), {
                      addSuffix: true,
                    })}
                  </p>
                </div>
                {!s.current && (
                  <button
                    onClick={() => terminateSession(s.sessionToken)}
                    disabled={deleteStatus.loading}
                    className="px-3 py-1 text-sm bg-red-100 text-red-700 hover:bg-red-200 dark:bg-red-900/50 dark:text-red-300 dark:hover:bg-red-800 rounded-md transition-colors"
                  >
                    {deleteStatus.loading ? "Terminating..." : "Terminate"}
                  </button>
                )}
              </li>
            ))
          )}
        </ul>
      </div>

      {sessions.filter((s) => !s.current).length > 0 && (
        <div className="mt-4 flex justify-end">
          <button
            onClick={terminateAllOtherSessions}
            disabled={deleteStatus.loading}
            className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-md transition-colors"
          >
            {deleteStatus.loading
              ? "Terminating..."
              : "Terminate All Other Sessions"}
          </button>
        </div>
      )}
    </div>
  );
}
