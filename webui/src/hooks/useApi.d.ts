import type { ApiResponse } from '@/types';
interface UseApiResult<T> {
    data: T | null;
    loading: boolean;
    error: string | null;
    refetch: () => void;
}
interface UseApiOptions {
    refreshInterval?: number;
    immediate?: boolean;
}
type UseApiSource<T> = string | (() => Promise<ApiResponse<T>>);
export declare function useApi<T>(source: UseApiSource<T>, options?: UseApiOptions): UseApiResult<T>;
export {};
