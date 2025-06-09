"use client";

import {
  signIn as nextAuthSignIn,
  signOut as nextAuthSignOut,
} from "next-auth/react";

export function signIn(provider, options) {
  return nextAuthSignIn(provider, options);
}

export function signOut(options) {
  return nextAuthSignOut(options);
}
