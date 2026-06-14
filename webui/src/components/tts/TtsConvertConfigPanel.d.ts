interface ConvertFormData {
    input_dir: string;
    output_dir: string;
    voice_id: string;
    concurrency: number;
    rate: string;
    recursive: boolean;
    resume: boolean;
}
interface TtsConvertConfigPanelProps {
    formData: ConvertFormData;
    setFormData: (data: ConvertFormData) => void;
    voiceOptions: {
        value: string;
        label: string;
    }[];
    voicesLoading: boolean;
    submitting: boolean;
    submitError: string | null;
    onSubmit: () => void;
}
export declare function TtsConvertConfigPanel({ formData, setFormData, voiceOptions, voicesLoading, submitting, submitError, onSubmit, }: TtsConvertConfigPanelProps): import("react/jsx-runtime").JSX.Element;
export {};
