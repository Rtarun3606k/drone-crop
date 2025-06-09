import React from "react";

// Dummy user data (replace with real data or props as needed)
const user = {
  name: "Jane Doe",
  email: "jane.doe@example.com",
  username: "janedoe",
  location: "San Francisco, CA",
  bio: "Drone enthusiast and crop monitoring specialist.",
  avatar:
    "https://ui-avatars.com/api/?name=Jane+Doe&background=0D8ABC&color=fff",
  joined: "2023-01-15",
};

export default function ProfilePage() {
  return (
    <div className="min-h-screen bg-transparent flex flex-col items-center py-10">
      <div className="bg-gray-900 shadow-lg rounded-lg p-8 w-full max-w-md border border-gray-800">
        <div className="flex flex-col items-center">
          <img
            src={user.avatar}
            alt="User avatar"
            className="w-24 h-24 rounded-full border-4 border-blue-600 mb-4"
          />
          <h2 className="text-2xl font-bold mb-1 text-white">{user.name}</h2>
          <p className="text-gray-400 mb-2">@{user.username}</p>
          <p className="text-gray-300 text-center mb-4">{user.bio}</p>
        </div>
        <div className="border-t border-gray-700 pt-4 mt-4 space-y-2">
          <div className="flex justify-between">
            <span className="font-medium text-gray-400">Email:</span>
            <span className="text-gray-200">{user.email}</span>
          </div>
          <div className="flex justify-between">
            <span className="font-medium text-gray-400">Location:</span>
            <span className="text-gray-200">{user.location}</span>
          </div>
          <div className="flex justify-between">
            <span className="font-medium text-gray-400">Joined:</span>
            <span className="text-gray-200">
              {new Date(user.joined).toLocaleDateString()}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}