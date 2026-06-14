interface ConfigResetDialogProps {
    open: boolean;
    onClose: () => void;
    onReset: (preset: string) => Promise<void>;
    resetting: boolean;
}
export declare function ConfigResetDialog({ open, onClose, onReset, resetting, }: ConfigResetDialogProps): import("react/jsx-runtime").JSX.Element;
export {};
