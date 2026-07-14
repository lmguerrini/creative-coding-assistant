import { useId, type ReactNode } from "react";
import type { LucideIcon } from "lucide-react";

function joinClassNames(...values: Array<string | undefined>) {
  return values.filter(Boolean).join(" ");
}

export type DashboardPageHeroProps = {
  badgeLabel: string;
  badges: readonly string[];
  className?: string;
  detail: string;
  eyebrow: string;
  headingLevel?: "h1" | "h2";
  icon: LucideIcon;
  title: string;
  tone?: string;
};

export function DashboardPage({
  children,
  className,
  hero,
  label
}: {
  children: ReactNode;
  className?: string;
  hero: DashboardPageHeroProps;
  label: string;
}) {
  return (
    <section aria-label={label} className={joinClassNames("dashboardPage", className)}>
      <DashboardPageHero {...hero} />
      {children}
    </section>
  );
}

export function DashboardPageHero({
  badgeLabel,
  badges,
  className,
  detail,
  eyebrow,
  headingLevel = "h2",
  icon: Icon,
  title,
  tone
}: DashboardPageHeroProps) {
  const Heading = headingLevel;

  return (
    <header className={joinClassNames("dashboardPageHero", className)} data-tone={tone}>
      <div className="dashboardPageHeroIcon" aria-hidden="true">
        <Icon size={24} />
      </div>
      <div className="dashboardPageHeroCopy">
        <span>{eyebrow}</span>
        <Heading>{title}</Heading>
        <p>{detail}</p>
      </div>
      <div className="dashboardPageHeroBadges" aria-label={badgeLabel} role="list">
        {badges.map((badge) => <span key={badge} role="listitem">{badge}</span>)}
      </div>
    </header>
  );
}

export function DashboardSection({
  action,
  as = "section",
  children,
  className,
  detail,
  eyebrow,
  icon,
  id,
  label,
  state,
  title,
  titleId,
  tone
}: {
  action?: ReactNode;
  as?: "article" | "section";
  children: ReactNode;
  className?: string;
  detail: string;
  eyebrow?: string;
  icon: LucideIcon;
  id?: string;
  label?: string;
  state?: string;
  title: string;
  titleId?: string;
  tone?: string;
}) {
  const Element = as;
  return (
    <Element
      aria-label={label}
      aria-labelledby={titleId}
      className={joinClassNames("dashboardVisualSection", className)}
      data-state={state}
      data-tone={tone}
      id={id}
    >
      <DashboardSectionHeader
        action={action}
        detail={detail}
        eyebrow={eyebrow}
        icon={icon}
        title={title}
        titleId={titleId}
      />
      {children}
    </Element>
  );
}

export function DashboardSectionHeader({
  action,
  className,
  detail,
  eyebrow,
  icon: Icon,
  title,
  titleId
}: {
  action?: ReactNode;
  className?: string;
  detail: string;
  eyebrow?: string;
  icon: LucideIcon;
  title: string;
  titleId?: string;
}) {
  return (
    <header className={joinClassNames("dashboardSectionHeader", className)}>
      <div className="dashboardSectionHeaderIcon" aria-hidden="true">
        <Icon size={19} />
      </div>
      <div className="dashboardSectionHeaderCopy">
        {eyebrow ? <span>{eyebrow}</span> : null}
        <h2 id={titleId}>{title}</h2>
        <p>{detail}</p>
      </div>
      {action ? <div className="dashboardSectionHeaderAction">{action}</div> : null}
    </header>
  );
}

export function DashboardCardGrid({
  children,
  className,
  label,
  layout = "auto",
  role
}: {
  children: ReactNode;
  className?: string;
  label?: string;
  layout?: "auto" | "compact" | "equal" | "quad" | "steps";
  role?: "group" | "list";
}) {
  return (
    <div
      aria-label={label}
      className={joinClassNames("dashboardCardGrid", className)}
      data-layout={layout}
      role={role}
    >
      {children}
    </div>
  );
}

export function DashboardInfoCard({
  children,
  className,
  detail,
  eyebrow,
  icon: Icon,
  label,
  role,
  selected,
  title,
  tone
}: {
  children?: ReactNode;
  className?: string;
  detail: string;
  eyebrow?: string;
  icon?: LucideIcon;
  label?: string;
  role?: "group" | "listitem";
  selected?: boolean;
  title: string;
  tone?: string;
}) {
  return (
    <article
      aria-label={label}
      className={joinClassNames("dashboardInnerCard", "dashboardInfoCard", className)}
      data-selected={selected ? "true" : undefined}
      data-tone={tone}
      role={role}
    >
      {Icon ? <Icon aria-hidden="true" size={19} /> : null}
      {eyebrow ? <span>{eyebrow}</span> : null}
      <strong>{title}</strong>
      <p>{detail}</p>
      {children}
    </article>
  );
}

export function DashboardActionCard({
  badge,
  className,
  detail,
  icon: Icon,
  onClick,
  title,
  tone
}: {
  badge?: string;
  className?: string;
  detail: string;
  icon?: LucideIcon;
  onClick: () => void;
  title: string;
  tone?: string;
}) {
  const cardId = useId();
  const titleId = `${cardId}-title`;
  const detailId = `${cardId}-detail`;
  const badgeId = badge ? `${cardId}-badge` : undefined;

  return (
    <button
      aria-describedby={badgeId ? `${badgeId} ${detailId}` : detailId}
      aria-labelledby={titleId}
      className={joinClassNames("dashboardInnerCard", "dashboardActionCard", className)}
      data-has-icon={Icon ? "true" : "false"}
      data-tone={tone}
      onClick={onClick}
      type="button"
    >
      {Icon ? (
        <span className="dashboardActionCardIcon" aria-hidden="true">
          <Icon size={19} />
        </span>
      ) : null}
      {badge ? <span className="dashboardActionCardBadge" id={badgeId}>{badge}</span> : null}
      <span className="dashboardActionCardTitle" id={titleId}>{title}</span>
      <span className="dashboardActionCardDetail" id={detailId}>{detail}</span>
      <span className="dashboardActionCardCue" aria-hidden="true">Use this starter →</span>
    </button>
  );
}

export function DashboardChoiceCard({
  ariaCurrent,
  ariaPressed = true,
  className,
  detail,
  eyebrow,
  idleLabel = "View",
  icon: Icon,
  onClick,
  selected,
  selectedLabel = "Selected",
  title
}: {
  ariaCurrent?: "page" | "true";
  ariaPressed?: boolean;
  className?: string;
  detail: string;
  eyebrow: string;
  idleLabel?: string;
  icon?: LucideIcon;
  onClick: () => void;
  selected: boolean;
  selectedLabel?: string;
  title: string;
}) {
  const cardId = useId();
  const titleId = `${cardId}-title`;
  const eyebrowId = `${cardId}-eyebrow`;
  const descriptionId = `${cardId}-description`;

  return (
    <button
      aria-current={ariaCurrent}
      aria-describedby={`${eyebrowId} ${descriptionId}`}
      aria-labelledby={titleId}
      aria-pressed={ariaPressed ? selected : undefined}
      className={joinClassNames("dashboardInnerCard", "dashboardChoiceCard", className)}
      data-has-icon={Icon ? "true" : "false"}
      data-selected={selected ? "true" : undefined}
      onClick={onClick}
      type="button"
    >
      {Icon ? (
        <span aria-hidden="true" className="dashboardChoiceCardIcon">
          <Icon size={18} />
        </span>
      ) : null}
      <span className="dashboardChoiceCardEyebrow" id={eyebrowId}>{eyebrow}</span>
      <span aria-hidden="true" className="dashboardChoiceCardState">
        {selected ? selectedLabel : idleLabel}
      </span>
      <span className="dashboardChoiceCardTitle" id={titleId}>{title}</span>
      <span className="dashboardChoiceCardDetail" id={descriptionId}>{detail}</span>
    </button>
  );
}

export function DashboardSidebarHeader({
  action,
  className,
  detail,
  eyebrow,
  icon: Icon,
  title,
  titleAs = "strong"
}: {
  action?: ReactNode;
  className?: string;
  detail: string;
  eyebrow: string;
  icon: LucideIcon;
  title: string;
  titleAs?: "h2" | "strong";
}) {
  const Title = titleAs;

  return (
    <header className={joinClassNames("dashboardSidebarHeader", className)}>
      <span aria-hidden="true" className="dashboardSidebarHeaderIcon">
        <Icon size={19} />
      </span>
      <div className="dashboardSidebarHeaderCopy">
        <span>{eyebrow}</span>
        <Title>{title}</Title>
        <p>{detail}</p>
      </div>
      {action ? <div className="dashboardSidebarHeaderAction">{action}</div> : null}
    </header>
  );
}

export function DashboardDefinitionGrid({
  className,
  items,
  label,
  layout = "auto"
}: {
  className?: string;
  items: readonly { label: string; value: ReactNode }[];
  label: string;
  layout?: "auto" | "compact" | "wide";
}) {
  return (
    <dl
      aria-label={label}
      className={joinClassNames("dashboardDefinitionGrid", className)}
      data-layout={layout}
    >
      {items.map((item) => (
        <div className="dashboardInnerCard dashboardDefinitionCard" key={item.label}>
          <dt>{item.label}</dt>
          <dd>{item.value}</dd>
        </div>
      ))}
    </dl>
  );
}

export function DashboardMetricGrid({
  className,
  label,
  metrics
}: {
  className?: string;
  label: string;
  metrics: readonly {
    detail?: string;
    label: string;
    tone?: string;
    value: ReactNode;
  }[];
}) {
  return (
    <dl aria-label={label} className={joinClassNames("dashboardMetricGrid", className)}>
      {metrics.map((metric) => (
        <div className="dashboardInnerCard dashboardMetricCard" data-tone={metric.tone} key={metric.label}>
          <dt>{metric.label}</dt>
          <dd>{metric.value}</dd>
          {metric.detail ? <small>{metric.detail}</small> : null}
        </div>
      ))}
    </dl>
  );
}

export function DashboardProcessRail({
  className,
  connectors = false,
  label,
  steps,
  variant = "compact"
}: {
  className?: string;
  connectors?: boolean;
  label: string;
  steps: readonly { detail: string; icon?: LucideIcon; title: string }[];
  variant?: "compact" | "journey";
}) {
  return (
    <ol
      aria-label={label}
      className={joinClassNames("dashboardProcessRail", className)}
      data-variant={variant}
    >
      {steps.map((step, index) => (
        <li key={step.title}>
          <span aria-hidden="true" className="dashboardProcessRailMarker">{index + 1}</span>
          {step.icon ? <step.icon aria-hidden="true" className="dashboardProcessRailIcon" size={18} /> : null}
          <div className="dashboardProcessRailCopy">
            <strong>{step.title}</strong>
            <small>{step.detail}</small>
          </div>
          {connectors && index < steps.length - 1 ? (
            <span aria-hidden="true" className="dashboardProcessRailConnector">›</span>
          ) : null}
        </li>
      ))}
    </ol>
  );
}

export function DashboardCallout({
  as = "aside",
  children,
  className,
  detail,
  icon: Icon,
  title,
  tone
}: {
  as?: "aside" | "footer";
  children?: ReactNode;
  className?: string;
  detail: string;
  icon?: LucideIcon;
  title: string;
  tone?: "info" | "success" | "warning";
}) {
  const Element = as;
  return (
    <Element
      className={joinClassNames("dashboardCallout", className)}
      data-has-icon={Icon ? "true" : "false"}
      data-tone={tone}
    >
      {Icon ? <Icon aria-hidden="true" size={19} /> : null}
      <div>
        <strong>{title}</strong>
        <p>{detail}</p>
        {children}
      </div>
    </Element>
  );
}

export function DashboardDisclosure({
  children,
  className,
  defaultOpen = false,
  summary,
  summaryLabel
}: {
  children: ReactNode;
  className?: string;
  defaultOpen?: boolean;
  summary: ReactNode;
  summaryLabel?: string;
}) {
  return (
    <details className={joinClassNames("dashboardDisclosure", className)} open={defaultOpen || undefined}>
      <summary aria-label={summaryLabel}>{summary}</summary>
      <div className="dashboardDisclosureBody">{children}</div>
    </details>
  );
}

export function DashboardTableFrame({
  children,
  className
}: {
  children: ReactNode;
  className?: string;
}) {
  return <div className={joinClassNames("dashboardTableFrame", className)}>{children}</div>;
}

export function DashboardTabs({
  className,
  items,
  label
}: {
  className?: string;
  items: readonly { href: string; label: string }[];
  label: string;
}) {
  return (
    <nav aria-label={label} className={joinClassNames("dashboardTabs", className)}>
      {items.map((item) => <a href={item.href} key={item.href}>{item.label}</a>)}
    </nav>
  );
}
