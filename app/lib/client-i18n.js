"use client";

import { useTranslations as useNextIntlTranslations } from "next-intl";

// This is a wrapper to handle cases where the context might not be available yet
export function useTranslations(namespace) {
  try {
    return useNextIntlTranslations(namespace);
  } catch (error) {
    // Fallback function for when context is not available
    return (key) => {
      const fallbackTranslations = {
        common: {
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
