"use client";

import SwaggerUI from "swagger-ui-react";
import "swagger-ui-react/swagger-ui.css";

function ReactSwagger({ spec }) {
  return (
    <SwaggerUI spec={spec} docExpansion="list" defaultModelsExpandDepth={-1} />
  );
}

export default ReactSwagger;
