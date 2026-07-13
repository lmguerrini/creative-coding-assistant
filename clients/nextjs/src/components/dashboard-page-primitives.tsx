import type { ReactNode } from "react";
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
  icon: Icon,
  title,
  tone
}: DashboardPageHeroProps) {
  return (
    <header className={joinClassNames("dashboardPageHero", className)} data-tone={tone}>
      <div className="dashboardPageHeroIcon" aria-hidden="true">
        <Icon size={24} />
      </div>
      <div className="dashboardPageHeroCopy">
        <span>{eyebrow}</span>
        <h2>{title}</h2>
        <p>{detail}</p>
      </div>
      <div className="dashboardPageHeroBadges" aria-label={badgeLabel}>
        {badges.map((badge) => <span key={badge}>{badge}</span>)}
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
  layout?: "auto" | "compact" | "equal" | "steps";
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
  icon: LucideIcon;
  title: string;
  tone?: "info" | "success" | "warning";
}) {
  const Element = as;
  return (
    <Element className={joinClassNames("dashboardCallout", className)} data-tone={tone}>
      <Icon aria-hidden="true" size={19} />
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
