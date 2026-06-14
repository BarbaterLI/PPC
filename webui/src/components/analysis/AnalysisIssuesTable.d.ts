interface AnalysisIssueItem {
    analyzer: string;
    severity: string;
    message: string;
    suggestion?: string;
}
interface AnalysisIssuesTableProps {
    issues: AnalysisIssueItem[];
}
export declare function AnalysisIssuesTable({ issues, }: AnalysisIssuesTableProps): import("react/jsx-runtime").JSX.Element;
export {};
