"use client";

import {
  getProductIntelligenceSection,
  productIntelligenceCategories,
  type ProductIntelligenceCategory,
  type ProductIntelligenceModel,
  type ProductIntelligenceSection
} from "@/lib/product-intelligence";

type ProductIntelligenceDashboardProps = {
  activeCategory: ProductIntelligenceCategory;
  model: ProductIntelligenceModel;
  onCategoryChange: (category: ProductIntelligenceCategory) => void;
  onClose: () => void;
};

export function ProductIntelligenceDashboard({
  activeCategory,
  model,
  onCategoryChange,
  onClose
}: ProductIntelligenceDashboardProps) {
  const section = getProductIntelligenceSection(model, activeCategory);

  return (
    <section aria-label="Product Intelligence Dashboard" className="productDashboard">
      <nav aria-label="Dashboard categories" className="productDashboardNav">
        <header>
          <span>Dashboard</span>
          <strong>Product Intelligence</strong>
          <p>Detailed product state, kept separate from the creative workspace.</p>
        </header>
        <div role="list">
          {productIntelligenceCategories.map((category) => {
            const item = getProductIntelligenceSection(model, category);
            return (
              <button
                aria-current={category === activeCategory ? "page" : undefined}
                data-tone={item.tone}
                key={category}
                onClick={() => onCategoryChange(category)}
                type="button"
              >
                <span>{category}</span>
                <small>{item.summary}</small>
              </button>
            );
          })}
        </div>
      </nav>
      <div className="productDashboardContent">
        <header className="productDashboardContentHeader">
          <div>
            <span>Dashboard category</span>
            <h1>{section.category}</h1>
            <p>{section.detail}</p>
          </div>
          <div className="productDashboardStatus" data-tone={section.tone}>
            {section.summary}
          </div>
          <button aria-label="Return to workspace" onClick={onClose} type="button">
            Return to workspace
          </button>
        </header>
        <ProductIntelligenceSectionView detailed section={section} />
      </div>
    </section>
  );
}

export function ProductIntelligenceInspector({
  category,
  model
}: {
  category: ProductIntelligenceCategory;
  model: ProductIntelligenceModel;
}) {
  return (
    <section
      aria-label={`${category} inspector`}
      className="inspectorPanel productIntelligenceInspector"
      id={`${category.toLowerCase().replace(/\s+/g, "-")}-inspector-panel`}
      role="tabpanel"
    >
      <ProductIntelligenceSectionView
        section={getProductIntelligenceSection(model, category)}
      />
    </section>
  );
}

function ProductIntelligenceSectionView({
  detailed = false,
  section
}: {
  detailed?: boolean;
  section: ProductIntelligenceSection;
}) {
  const notes = detailed ? section.notes : section.notes.slice(0, 2);
  const metrics = detailed ? section.metrics : section.metrics.slice(0, 3);

  return (
    <div className="productIntelligenceSection" data-tone={section.tone}>
      <article className="productIntelligenceHero">
        <span>{section.category}</span>
        <strong>{section.summary}</strong>
        <p>{section.detail}</p>
      </article>
      <dl className="productIntelligenceMetrics">
        {metrics.map((metric) => (
          <div key={metric.label}>
            <dt>{metric.label}</dt>
            <dd title={metric.value}>{metric.value}</dd>
          </div>
        ))}
      </dl>
      <div className="productIntelligenceNotes" aria-label={`${section.category} details`}>
        {notes.map((note) => (
          <p key={note}>{note}</p>
        ))}
      </div>
    </div>
  );
}
