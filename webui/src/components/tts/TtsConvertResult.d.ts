interface FailedItem {
    file: string;
    error: string;
}
interface ConvertResult {
    success_count: number;
    failure_count: number;
    duration: number;
    failed_items: FailedItem[];
}
interface TtsConvertResultProps {
    result: ConvertResult;
    onReset: () => void;
}
export declare function TtsConvertResult({ result, onReset, }: TtsConvertResultProps): import("react/jsx-runtime").JSX.Element;
export {};
