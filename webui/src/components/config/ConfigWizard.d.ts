interface WizardData {
    voice: string;
    concurrency: number;
    outputFormat: string;
}
interface ConfigWizardProps {
    open: boolean;
    onClose: () => void;
    onComplete: (data: WizardData) => void;
}
export declare function ConfigWizard({ open, onClose, onComplete, }: ConfigWizardProps): import("react/jsx-runtime").JSX.Element;
export {};
