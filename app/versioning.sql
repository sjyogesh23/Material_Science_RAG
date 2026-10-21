-- Versioned components (FEATURES_V2 §6). Idempotent; applied by app.versioning.ensure_schema().
CREATE TABLE IF NOT EXISTS component_versions (
    id          bigserial PRIMARY KEY,
    component   text NOT NULL,
    version     int  NOT NULL,
    content     jsonb NOT NULL,
    note        text,
    created_at  timestamptz DEFAULT now(),
    created_by  text DEFAULT 'ui',          -- ui | code (seeded from code defaults) | <author>
    is_active   bool DEFAULT false,
    UNIQUE (component, version)
);
-- at most one active version per component
CREATE UNIQUE INDEX IF NOT EXISTS component_versions_one_active ON component_versions (component) WHERE is_active;
CREATE INDEX IF NOT EXISTS component_versions_component ON component_versions (component, version DESC);
