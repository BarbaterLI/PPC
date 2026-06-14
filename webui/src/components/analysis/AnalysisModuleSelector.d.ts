interface AnalysisModuleSelectorProps {
    selectedModules: Set<string>;
    onToggleModule: (key: string) => void;
    onStartAnalysis: () => void;
    onStopAnalysis: () => void;
    onToggleHistory: () => void;
    analyzing: boolean;
}
export declare function AnalysisModuleSelector({ selectedModules, onToggleModule, onStartAnalysis, onStopAnalysis, onToggleHistory, analyzing, }: AnalysisModuleSelectorProps): import("react/jsx-runtime").JSX.Element;
export {};
