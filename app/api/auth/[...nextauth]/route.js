import { handlers } from "../../../auth"; // Referring to the auth.js file at app/auth.js

// Force Node.js runtime for auth routes
export const runtime = "nodejs";
export const dynamic = "force-dynamic"; // Ensures the route is not statically optimized
export const preferredRegion = "auto";

export const { GET, POST } = handlers;
