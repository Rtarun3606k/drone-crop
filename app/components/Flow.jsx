import React from "react";
import { motion } from "framer-motion";
import {
  FiUploadCloud,
  FiImage,
  FiCpu,
  FiMessageSquare,
  FiGlobe,
  FiVolume2,
  FiSend
} from "react-icons/fi";
import { useTranslations } from "next-intl";



export const Flow = () => {
  const t = useTranslations("Flow");
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.3,
        delayChildren: 0.2
      }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, x: -50, scale: 0.8 },
    visible: {
      opacity: 1,
      x: 0,
      scale: 1,
      transition: { duration: 0.6, ease: "easeOut" }
    }
  };

  const iconVariants = {
    hidden: { scale: 0, rotate: -180 },
    visible: {
      scale: 1,
      rotate: 0,
      transition: { duration: 0.5, ease: "easeOut", delay: 0.3 }
    }
  };

  const lineVariants = {
    hidden: { scaleY: 0, originY: 0 },
    visible: {
      scaleY: 1,
      transition: { duration: 0.5, ease: "easeInOut" }
    }
  };

  // Fix: steps should be an array, not a string from translation
  const steps = [
    {
      title: t("steps.0.title"),
      text: t("steps.0.text"),
      icon: <FiUploadCloud size={32} className="text-green-600" />, side: "start"
    },
    {
      title: t("steps.1.title"),
      text: t("steps.1.text"),
      icon: <FiImage size={32} className="text-green-600" />, side: "end"
    },
    {
      title: t("steps.2.title"),
      text: t("steps.2.text"),
      icon: <FiCpu size={32} className="text-green-600" />, side: "start"
    },
    {
      title: t("steps.3.title"),
      text: t("steps.3.text"),
      icon: <FiMessageSquare size={32} className="text-green-600" />, side: "end"
    },
    {
      title: t("steps.4.title"),
      text: t("steps.4.text"),
      icon: <FiGlobe size={32} className="text-green-600" />, side: "start"
    },
    {
      title: t("steps.5.title"),
      text: t("steps.5.text"),
      icon: <FiVolume2 size={32} className="text-green-600" />, side: "end"
    },
    {
      title: t("steps.6.title"),
      text: t("steps.6.text"),
      icon: <FiSend size={32} className="text-green-600" />, side: "start"
    }
  ];

  return (
    <div className="px-6 py-10">
      <motion.ul
        className="timeline timeline-vertical"
        variants={containerVariants}
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true, amount: 0.2 }}
      >
        {steps.map((step, index) => (
          <motion.li key={index} variants={itemVariants}>
            {index !== 0 && (
              <motion.hr className="bg-green-500" variants={lineVariants} />
            )}

            {step.side === "start" ? (
              <motion.div
                className="timeline-start timeline-box bg-base-200 text-left shadow-xl max-w-xl"
                whileHover={{ scale: 1.05 }}
                transition={{ duration: 0.2 }}
              >
                <h3 className="text-lg font-bold mb-1">{step.title}</h3>
                <p className="text-base">{step.text}</p>
              </motion.div>
            ) : (
              <motion.div
                className="timeline-end timeline-box bg-base-200 text-left shadow-xl max-w-xl"
                whileHover={{ scale: 1.05 }}
                transition={{ duration: 0.2 }}
              >
                <h3 className="text-lg font-bold mb-1">{step.title}</h3>
                <p className="text-base">{step.text}</p>
              </motion.div>
            )}

            <div className="timeline-middle">
              <motion.div variants={iconVariants}>{step.icon}</motion.div>
            </div>
          </motion.li>
        ))}
      </motion.ul>
    </div>
  );
};
