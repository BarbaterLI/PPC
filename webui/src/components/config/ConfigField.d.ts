import type { PPC10Config } from '@/types';
interface ConfigFieldProps {
    config: PPC10Config;
    value: string;
    isChanged: boolean;
    onChange: (value: string) => void;
}
export declare function ConfigField({ config, value, isChanged, onChange, }: ConfigFieldProps): import("react/jsx-runtime").JSX.Element;
export {};
