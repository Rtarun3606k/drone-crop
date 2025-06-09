import { prisma } from "../lib/prisma-server";

/**
 * Get a user by their email address
 */
export async function getUserByEmail(email) {
  return prisma.user.findUnique({
    where: { email },
  });
}

/**
 * Get a user by their ID
 */
export async function getUserById(id) {
  return prisma.user.findUnique({
    where: { id },
  });
}

/**
 * Get user profile with additional information
 */
export async function getUserProfile(id) {
  return prisma.user.findUnique({
    where: { id },
    select: {
      id: true,
      name: true,
      email: true,
      image: true,
      createdAt: true,
      role: true, // Include user role
      accounts: {
        select: {
          provider: true,
        },
      },
      // Add any additional profile info you want to retrieve
    },
  });
}

/**
 * Update user profile information
 */
export async function updateUserProfile(id, data) {
  return prisma.user.update({
    where: { id },
    data: {
      name: data.name,
      image: data.image,
      // Add other fields you want to be updatable
    },
  });
}

/**
 * Get all active sessions for a user
 */
export async function getUserSessions(userId) {
  return prisma.session.findMany({
    where: { userId },
    orderBy: { expires: "desc" },
  });
}

/**
 * Delete a specific session
 */
export async function deleteSession(sessionToken) {
  return prisma.session.delete({
    where: { sessionToken },
  });
}

/**
 * Delete all sessions for a user except the current one
 */
export async function deleteOtherSessions(userId, currentSessionToken) {
  return prisma.session.deleteMany({
    where: {
      userId,
      NOT: {
        sessionToken: currentSessionToken,
      },
    },
  });
}

/**
 * Check if a user has admin role
 */
export async function isAdmin(userId) {
  const user = await prisma.user.findUnique({
    where: { id: userId },
    select: { role: true },
  });

  return user?.role === "ADMIN";
}
