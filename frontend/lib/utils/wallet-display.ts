/**
 * Utility functions for displaying wallet information with personality
 */

interface WalletData {
    proxy_wallet: string;
    pseudonym?: string | null;
    name?: string | null;
    profile_image?: string | null;
    bio?: string | null;
}

/**
 * Get display name for a wallet with fallback priority:
 * 1. pseudonym
 * 2. name
 * 3. truncated wallet address
 */
export function getWalletDisplayName(wallet: WalletData | null | undefined): string {
    if (!wallet) return 'Unknown';

    if (wallet.pseudonym) return wallet.pseudonym;
    if (wallet.name) return wallet.name;

    return formatWalletAddress(wallet.proxy_wallet);
}

/**
 * Get avatar URL for a wallet, returns null if no profile image
 */
export function getWalletAvatar(wallet: WalletData | null | undefined): string | null {
    if (!wallet) return null;
    return wallet.profile_image || null;
}

/**
 * Format wallet address to show first 4 and last 4 characters
 * e.g., 0x1234...5678
 */
export function formatWalletAddress(address: string): string {
    if (!address || address.length < 10) return address;
    return `${address.slice(0, 6)}...${address.slice(-4)}`;
}

/**
 * Check if wallet has personality data (pseudonym or name)
 */
export function hasWalletPersonality(wallet: WalletData | null | undefined): boolean {
    if (!wallet) return false;
    return !!(wallet.pseudonym || wallet.name);
}
