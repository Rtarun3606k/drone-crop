import React from "react";
import Link from "next/link";
import { FiHome, FiUser, FiBriefcase, FiInfo, FiMail } from "react-icons/fi";
import { getTranslations } from "next-intl/server";

const Footer = async () => {
  const t = await getTranslations("common");
  return (
    <footer
      style={{
        background: "#222",
        color: "#fff",
        padding: "2.5rem 0",
        textAlign: "center",
        marginTop: "auto",
      }}
    >
      <div style={{ marginBottom: "2rem" }}>
        <h2 style={{ margin: 0, fontSize: "1.7rem", letterSpacing: "1px" }}>
          {t("brand")}
        </h2>
      </div>
      <nav
        className="flex flex-col items-center gap-4 sm:flex-row sm:justify-center sm:gap-10"
        style={{ marginBottom: "2rem" }}
      >
        <Link
          href="/"
          className="flex items-center gap-2 text-white hover:text-green-400 transition-colors text-lg"
        >
          <FiHome size={22} />
          <span>{t("home")}</span>
        </Link>
        <Link
          href="/profile"
          className="flex items-center gap-2 text-white hover:text-green-400 transition-colors text-lg"
        >
          <FiUser size={22} />
          <span>{t("profile")}</span>
        </Link>
        <Link
          href="/services"
          className="flex items-center gap-2 text-white hover:text-green-400 transition-colors text-lg"
        >
          <FiBriefcase size={22} />
          <span>{t("services")}</span>
        </Link>
        <Link
          href="/about"
          className="flex items-center gap-2 text-white hover:text-green-400 transition-colors text-lg"
        >
          <FiInfo size={22} />
          <span>{t("about")}</span>
        </Link>
        <Link
          href="/contact"
          className="flex items-center gap-2 text-white hover:text-green-400 transition-colors text-lg"
        >
          <FiMail size={22} />
          <span>{t("contact")}</span>
        </Link>
      </nav>
      <div style={{ marginTop: "1.5rem" }}>
        <p style={{ margin: 0, fontSize: "1rem" }}>
          {t("contact")}:{" "}
          <a href="mailto:info@dronecrop.com" style={{ color: "#4caf50" }}>
            info@dronecrop.com
          </a>{" "}
          | +1 (555) 123-4567
        </p>
        <p
          style={{
            margin: "1rem 0 0 0",
            fontSize: "0.95rem",
            color: "#aaa",
          }}
        >
          &copy; {new Date().getFullYear()} {t("brand")}. {t("rights")}
        </p>
      </div>
    </footer>
  );
};

export default Footer;
