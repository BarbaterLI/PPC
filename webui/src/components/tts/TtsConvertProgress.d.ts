interface SseProgressData {
    progress: number;
    current_file?: string;
    speed?: string;
    eta?: string;
    total_files?: number;
    completed_files?: number;
}
interface TtsConvertProgressProps {
    progressData: SseProgressData;
    onCancel: () => void;
}
export declare function TtsConvertProgress({ progressData, onCancel, }: TtsConvertProgressProps): import("react/jsx-runtime").JSX.Element;
export {};
