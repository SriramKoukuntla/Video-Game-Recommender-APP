import MetaMatchLogo from '../../assets/MetaMatchLogo.png';
import './Header.css';

export const Header = () => (
    <div className="header">
        <img src={MetaMatchLogo} alt="MetaMatch Logo" className="MetaMatchLogo" />
        <h1>Meta Match</h1>
    </div>
);

