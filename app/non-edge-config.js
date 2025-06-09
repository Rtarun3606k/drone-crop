/**
 * This configuration is used for routes that should explicitly run on Node.js
 * rather than the Edge runtime, especially those that use ORM features
 * that don't work in Edge environments.
 */

export const authConfig = {
  runtime: "nodejs",
  preferredRegion: "auto",
};
