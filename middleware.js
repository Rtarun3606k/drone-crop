import { auth } from "./app/auth";

// Only run the middleware on specific routes that aren't auth-related
export const config = {
  matcher: [
    // Apply to all routes except those starting with:
    "/((?!api/auth|_next/static|_next/image|favicon.ico|public/|fonts/).*)",
  ],
};

// Export the middleware function with a safety check
export async function middleware(request) {
  return auth(request);
}
