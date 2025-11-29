int naiveSolution(int *nums, int n, int k)
{
        int n_sub_windows = n - 3 + 1;
        int min_incr = 0;

        for (int i = 0; i < n_sub_windows; ++i)
        {
                int *sub_window = &nums[i];
                int maxI = sub_window[0] >= sub_window[1]
                               ? (2 * sub_window[0] <= sub_window[2])
                               : (1 + sub_window[1] <= sub_window[2]);
                int incr = k >= sub_window[maxI] ? k - sub_window[maxI] : 0;
                min_incr += incr;
                sub_window[maxI] += incr;
        }
}
