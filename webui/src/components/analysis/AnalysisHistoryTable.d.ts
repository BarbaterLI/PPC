interface AnalysisHistoryItem {
    id: string;
    taskId?: string;
    date: string;
    score: number;
    analyzers: string[];
    summary?: string;
}
interface AnalysisHistoryTableProps {
    historyData: AnalysisHistoryItem[] | null;
    historyLoading: boolean;
    onViewHistoryItem: (item: AnalysisHistoryItem) => void;
}
export declare function AnalysisHistoryTable({ historyData, historyLoading, onViewHistoryItem, }: AnalysisHistoryTableProps): import("react/jsx-runtime").JSX.Element;
export {};
