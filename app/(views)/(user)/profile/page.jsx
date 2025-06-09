"use client";

import React from "react";
import { useSession } from "next-auth/react";

export default function ProfilePage() {
  const { data: session } = useSession();

  if (!session) {
    return <div className="text-white">Not signed in</div>;
  }

  const user = session.user;

  return (
    <div className="min-h-screen bg-transparent flex flex-col items-center py-10">
      <div className="bg-gray-900 shadow-lg rounded-lg p-8 w-full max-w-md border border-gray-800">
        <div className="flex flex-col items-center">
          <img
            src={user.image}
            alt="User avatar"
            className="w-24 h-24 rounded-full border-4 border-blue-600 mb-4"
          />
          <h2 className="text-2xl font-bold mb-1 text-white">{user.name}</h2>
          <p className="text-gray-400 mb-2">{user.email}</p>
        </div>
      </div>
    </div>
  );
}