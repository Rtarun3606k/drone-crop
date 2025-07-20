import jwt from "jsonwebtoken";

export const createJwtToken = (user) => {
  const payload = {
    email: user.email,
    name: user.name,
    mobileId: user.mobileId,
    role: user.role, // Assuming user has a role field
    image: user.image, // Assuming user has an image field
    userId: user.id, // Assuming user has an id field
  };

  const secretKey = process.env.JWT_SECRET || "your-secret-key";
  // console.log("JWT Secret Key:", secretKey); // Log the secret key for debugging
  const options = {
    expiresIn: "168h", // Token expiration time
    issuer: "drone-crop-api",
  };
  return jwt.sign(payload, secretKey, options);
};

export const refreshTokenIfNeeded = (req, res, next) => {
  try {
    const user = req.user;

    // Check if token expires in less than 2 hours
    const now = Math.floor(Date.now() / 1000);
    const timeUntilExpiry = user.exp - now;
    console.log("Time until token expiry:", timeUntilExpiry);

    if (timeUntilExpiry < 7200) {
      console.log("Refreshing token for user:", user.email);
      // 2 hours = 7200 seconds
      const newToken = createJwtToken(user);
      res.setHeader("X-New-Token", newToken);
    }

    next();
  } catch (error) {
    // If refresh fails, just continue without setting new token
    next();
  }
};
