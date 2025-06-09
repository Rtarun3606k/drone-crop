/** @type {import('next').NextConfig} */
const nextConfig = {
  images: {
    remotePatterns: [
      {
        protocol: "https",
        hostname: "lh3.googleusercontent.com",
        pathname: "**",
      },
    ],
  },
  experimental: {
    serverActions: {
      bodySizeLimit: "2mb",
    },
  },
  // Important: This ensures auth routes don't run in the Edge runtime
  // where PrismaClient cannot function properly
  skipMiddlewareUrlNormalize: true,
  skipTrailingSlashRedirect: true,
};

export default nextConfig;
