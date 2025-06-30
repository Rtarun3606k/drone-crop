"use client";

import { useTranslations as useNextIntlTranslations } from "next-intl";

// This is a wrapper to handle cases where the context might not be available yet
export function useTranslations(namespace) {
  let isUsingFallback = false;

  try {
    // Try to use the official next-intl translations first
    const translator = useNextIntlTranslations(namespace);
    return translator;
  } catch (error) {
    // Fallback function for when context is not available
    isUsingFallback = true;
    return (key) => {
      const fallbackTranslations = {
        common: {
          // Using the English version as fallback
          brand: "DroneCrops",
          home: "Home",
          services: "Services",
          about: "About",
          contact: "Contact",
          login: "Log in",
          signup: "Sign up",
          signout: "Sign out",
          profile: "Your Profile",
          dashboard: "Dashboard",
        },
      };

      return fallbackTranslations[namespace]?.[key] || key;
    };
  }
}
