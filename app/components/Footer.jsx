import React from "react";
import Link from "next/link";
import { FiHome, FiUser, FiBriefcase, FiInfo, FiMail, FiFacebook, FiTwitter, FiInstagram, FiLinkedin } from "react-icons/fi";
import { getTranslations } from "next-intl/server";

const Footer = async ({ backgroundImage = null }) => {
  const t = await getTranslations("common");
  
  const footerStyle = {
    background: backgroundImage 
      ? `linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.8)), url(${backgroundImage})`
      : "#222",
    backgroundSize: "contain",
    backgroundPosition: "top",
    backgroundRepeat: "no-repeat",
    color: "#fff",
    padding: "2.5rem 0",
    marginTop: "auto",
  };

  return (
    <footer style={footerStyle}>
      <div className="max-w-6xl mx-auto px-4">
        {/* Main footer content */}
        <div className="flex flex-col lg:flex-row lg:justify-between lg:items-center gap-8 mb-8">
          {/* Company name - Left */}
          <div className="text-center lg:text-left">
            <h2 style={{ margin: 0, fontSize: "1.8rem", letterSpacing: "1px", fontWeight: "bold" }}>
              {t("brand")}
            </h2>
          </div>

          {/* Navigation icons - Center */}
          <nav className="flex flex-wrap justify-center gap-6 lg:gap-8">
            <Link
              href="/"
              className="flex flex-col items-center gap-1 text-white hover:text-green-400 transition-colors group"
            >
              <FiHome size={24} className="group-hover:scale-110 transition-transform" />
              <span className="text-sm">{t("home")}</span>
            </Link>
            <Link
              href="/profile"
              className="flex flex-col items-center gap-1 text-white hover:text-green-400 transition-colors group"
            >
              <FiUser size={24} className="group-hover:scale-110 transition-transform" />
              <span className="text-sm">{t("profile")}</span>
            </Link>
            <Link
              href="/services"
              className="flex flex-col items-center gap-1 text-white hover:text-green-400 transition-colors group"
            >
              <FiBriefcase size={24} className="group-hover:scale-110 transition-transform" />
              <span className="text-sm">{t("services")}</span>
            </Link>
            <Link
              href="/about"
              className="flex flex-col items-center gap-1 text-white hover:text-green-400 transition-colors group"
            >
              <FiInfo size={24} className="group-hover:scale-110 transition-transform" />
              <span className="text-sm">{t("about")}</span>
            </Link>
            <Link
              href="/contact"
              className="flex flex-col items-center gap-1 text-white hover:text-green-400 transition-colors group"
            >
              <FiMail size={24} className="group-hover:scale-110 transition-transform" />
              <span className="text-sm">{t("contact")}</span>
            </Link>
          </nav>

          {/* Social icons - Right */}
          <div className="flex justify-center lg:justify-end gap-4">
            <a
              href="https://facebook.com"
              target="_blank"
              rel="noopener noreferrer"
              className="p-2 rounded-full bg-black bg-opacity-50 hover:bg-green-500 transition-colors backdrop-blur-sm"
            >
              <FiFacebook size={20} />
            </a>
            <a
              href="https://twitter.com"
              target="_blank"
              rel="noopener noreferrer"
              className="p-2 rounded-full bg-black bg-opacity-50 hover:bg-green-500 transition-colors backdrop-blur-sm"
            >
              <FiTwitter size={20} />
            </a>
            <a
              href="https://instagram.com"
              target="_blank"
              rel="noopener noreferrer"
              className="p-2 rounded-full bg-black bg-opacity-50 hover:bg-green-500 transition-colors backdrop-blur-sm"
            >
              <FiInstagram size={20} />
            </a>
            <a
              href="https://linkedin.com"
              target="_blank"
              rel="noopener noreferrer"
              className="p-2 rounded-full bg-black bg-opacity-50 hover:bg-green-500 transition-colors backdrop-blur-sm"
            >
              <FiLinkedin size={20} />
            </a>
          </div>
        </div>

        {/* Copyright info - Bottom centered */}
        <div className="border-t border-gray-600 border-opacity-50 pt-6">
          <div className="text-center">
            <p className="mb-2 text-sm">
              {t("contact")}:{" "}
              <a href="mailto:info@dronecrop.com" className="text-green-400 hover:text-green-300 transition-colors">
                info@dronecrop.com
              </a>{" "}
              | +1 (555) 123-4567
            </p>
            <p className="text-sm text-gray-300">
              &copy; {new Date().getFullYear()} {t("brand")}. {t("rights")}
            </p>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
