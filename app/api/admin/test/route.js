import { NextResponse } from "next/server";
import { auth } from "@/app/auth";
import { isAdmin } from "@/app/lib/db";
import { authConfig } from "@/app/non-edge-config";

// Force Node.js runtime for this route
export const runtime = authConfig.runtime;
export const preferredRegion = authConfig.preferredRegion;

export async function GET() {
  const session = await auth();

  // Check if user is authenticated
  if (!session || !session.user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  // Check if user is an admin
  const admin = await isAdmin(session.user.id);
  if (!admin) {
    return NextResponse.json(
      { error: "Forbidden. Admin access required" },
      { status: 403 }
    );
  }

  // Admin-only data
  return NextResponse.json({
    message: "Admin access granted",
    adminData: {
      timestamp: new Date().toISOString(),
      userCount: 10, // Example data
      systemStatus: "Healthy",
    },
  });
}
