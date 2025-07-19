import jwt from "jsonwebtoken";

const JWT_SECRET =
  process.env.JWT_SECRET || "your-secret-key-change-this-in-production";

// Middleware to verify JWT token
export const verifyToken = (req, res, next) => {
  try {
    // Get token from Authorization header
    const authHeader = req.headers.authorization;

    if (!authHeader) {
      return res.status(401).json({
        error: "Unauthorized",
        message: "No authorization header provided.",
      });
    }

    // Check if it's a Bearer token
    const token = authHeader.startsWith("Bearer ")
      ? authHeader.slice(7, authHeader.length)
      : null;

    if (!token) {
      return res.status(401).json({
        error: "Unauthorized",
        message: "Invalid authorization format. Use Bearer token.",
      });
    }

    // Verify the token
    const decoded = jwt.verify(token, JWT_SECRET);

    // Add user info to request object
    req.user = decoded;

    // Continue to next middleware/route handler
    next();
  } catch (error) {
    if (error.name === "TokenExpiredError") {
      return res.status(401).json({
        error: "Unauthorized",
        message: "Token has expired.",
      });
    }

    if (error.name === "JsonWebTokenError") {
      return res.status(401).json({
        error: "Unauthorized",
        message: "Invalid token.",
      });
    }

    // Generic error
    console.error("Token verification error:", error);
    return res.status(401).json({
      error: "Unauthorized",
      message: "Token verification failed.",
    });
  }
};
