"use client";
import { Carousel } from "@/app/components/Carousel";
import { Flow } from "@/app/components/Flow";
import { Hero } from "@/app/components/Hero";
import { motion } from "framer-motion";
import { Link } from "@/i18n/routing";
import { useSession } from "next-auth/react";
import { useTranslations } from "next-intl";

import React from "react";

const page = () => {
  const { data: session } = useSession();

  const t = useTranslations("Home");
  return (
    <div>
      {/* <motion.div
        initial={{ opacity: 0, y: 40 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, amount: 0.3 }}
        transition={{ duration: 0.7, ease: [0.23, 1, 0.32, 1] }}
      >
      </motion.div> */}
        <Hero />
      <motion.div
        initial={{ opacity: 0, y: 40 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, amount: 0.3 }}
        transition={{ duration: 0.7, delay: 0.15, ease: [0.23, 1, 0.32, 1] }}
      >
        <Flow />
      </motion.div>
      <motion.div
        initial={{ opacity: 0, y: 40 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, amount: 0.3 }}
        transition={{ duration: 0.7, delay: 0.3, ease: [0.23, 1, 0.32, 1] }}
      >
        <Carousel />
      </motion.div>
    </div>
  );
};
export default page;
