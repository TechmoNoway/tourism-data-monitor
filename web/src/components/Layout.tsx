import { ReactNode } from 'react';
import Header from './Header';

interface LayoutProps {
  children: ReactNode;
}

const Layout = ({ children }: LayoutProps) => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      <Header />
      <main className="container mx-auto px-4 pt-24 pb-8">
        {children}
      </main>
      <footer className="bg-gradient-to-r from-gray-900 to-black text-white py-8 mt-12 border-t border-gray-800">
        <div className="container mx-auto px-4 text-center">
          <p className="text-gray-400 font-medium">
            © 2025 Tourism Monitor. Powered by AI Analytics.
          </p>
        </div>
      </footer>
    </div>
  );
};

export default Layout;
