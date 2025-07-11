import { Link } from "@/i18n/routing";
import React from "react";
import { FiUpload } from "react-icons/fi";
import { motion } from "framer-motion";
import { useTranslations } from "next-intl";


export const Hero = () => {
      const t = useTranslations("Home");

  return (
    <div>
      <div
        className="hero min-h-[50vh] relative"
        style={{
          backgroundImage:
            "url(https://www.innovationnewsnetwork.com/wp-content/uploads/2023/11/shutterstockLove-Silhouette_1304774914.jpg)",
        }}
      >
        <div className="hero-overlay"></div>
        <motion.div 
          className="hero-content text-neutral-content text-left justify-start"
          initial={{ opacity: 0, x: -100 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
        >
          <motion.div 
            className="max-w-[40%] ml-8 pl-8"
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2, ease: "easeOut" }}
          >
            <motion.h1 
              className="mb-5 text-5xl font-bold"
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.4, ease: "easeOut" }}
            >
              {t("title")}{" "}
            </motion.h1>
            <motion.p 
              className="mb-5"
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.6, ease: "easeOut" }}
            >
              {t("description")}
            </motion.p>
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.8, ease: "easeOut" }}
            >
              <Link href="/dashboard">
                <motion.button 
                  className="btn bg-[#00A63E] hover:bg-[#00a63dac]"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  transition={{ duration: 0.2 }}
                >
                  <FiUpload />
                  {t("actionButton")}
                </motion.button>
              </Link>
            </motion.div>
          </motion.div>
        </motion.div>
      </div>
    </div>
  );
};
