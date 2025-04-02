import UIKit

// MARK: - Data Structures

struct Move: Codable {
    let user: String
    let symbol: String
    let cell: Int
    let moveNumber: Int
}

struct GameResult: Codable {
    let winner: String?
    let moves: [Move]
    let date: Date
    let gameId: String
}

// MARK: - ViewController

class ViewController: UIViewController {
    
    // MARK: - Properties
    
    private var buttons: [UIButton] = []
    private var currentPlayer = "Stalin" // Stalin starts with X
    private var currentSymbol = "X"
    private var gameActive = true
    private var moveCount = 0
    private var moves: [Move] = []
    private var gameResults: [GameResult] = []
    
    // UI Elements
    private let statusLabel = UILabel()
    private let resetGameButton = UIButton()
    private let exportButton = UIButton()
    private let resetDataButton = UIButton()  // New button to clear saved JSON/CSV data
    
    // MARK: - Lifecycle
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupUI()
        loadSavedData()
    }
    
    // MARK: - UI Setup
    
    private func setupUI() {
        view.backgroundColor = .white
        
        // Title Label
        let titleLabel = UILabel()
        titleLabel.text = "Tic-Tac-Toe Data Collector"
        titleLabel.textAlignment = .center
        titleLabel.font = UIFont.boldSystemFont(ofSize: 22)
        titleLabel.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(titleLabel)
        
        // Grid Container using UIStackViews for a 3x3 grid
        let gridContainer = UIStackView()
        gridContainer.axis = .vertical
        gridContainer.distribution = .fillEqually
        gridContainer.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(gridContainer)
        
        // Create 3 rows, each containing 3 buttons
        for row in 0..<3 {
            let rowStack = UIStackView()
            rowStack.axis = .horizontal
            rowStack.distribution = .fillEqually
            rowStack.translatesAutoresizingMaskIntoConstraints = false
            
            for col in 0..<3 {
                let button = UIButton()
                // Set a darker background so white symbols are visible
                button.backgroundColor = .darkGray
                // Set white text for X and O
                button.setTitleColor(.white, for: .normal)
                button.titleLabel?.font = UIFont.systemFont(ofSize: 40, weight: .bold)
                button.layer.borderWidth = 1
                button.layer.borderColor = UIColor.lightGray.cgColor
                button.translatesAutoresizingMaskIntoConstraints = false
                button.tag = row * 3 + col  // Unique tag for each cell
                button.addTarget(self, action: #selector(buttonTapped), for: .touchUpInside)
                
                rowStack.addArrangedSubview(button)
                buttons.append(button)
            }
            gridContainer.addArrangedSubview(rowStack)
        }
        
        // Status Label to show game status and current turn
        statusLabel.text = "Player: Stalin (X)"
        statusLabel.textAlignment = .center
        statusLabel.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(statusLabel)
        
        // Reset Game Button (starts a new game)
        resetGameButton.setTitle("New Game", for: .normal)
        resetGameButton.backgroundColor = .systemBlue
        resetGameButton.setTitleColor(.white, for: .normal)
        resetGameButton.layer.cornerRadius = 8
        resetGameButton.translatesAutoresizingMaskIntoConstraints = false
        resetGameButton.addTarget(self, action: #selector(resetGame), for: .touchUpInside)
        view.addSubview(resetGameButton)
        
        // Export Button to share JSON/CSV data
        exportButton.setTitle("Export Data", for: .normal)
        exportButton.backgroundColor = .systemGreen
        exportButton.setTitleColor(.white, for: .normal)
        exportButton.layer.cornerRadius = 8
        exportButton.translatesAutoresizingMaskIntoConstraints = false
        exportButton.addTarget(self, action: #selector(exportData), for: .touchUpInside)
        view.addSubview(exportButton)
        
        // Reset Data Button to clear stored JSON and CSV files
        resetDataButton.setTitle("Reset Data", for: .normal)
        resetDataButton.backgroundColor = .systemRed
        resetDataButton.setTitleColor(.white, for: .normal)
        resetDataButton.layer.cornerRadius = 8
        resetDataButton.translatesAutoresizingMaskIntoConstraints = false
        resetDataButton.addTarget(self, action: #selector(resetData), for: .touchUpInside)
        view.addSubview(resetDataButton)
        
        // Layout Constraints
        NSLayoutConstraint.activate([
            // Title Label constraints
            titleLabel.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 20),
            titleLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 20),
            titleLabel.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -20),
            
            // Grid Container constraints
            gridContainer.topAnchor.constraint(equalTo: titleLabel.bottomAnchor, constant: 40),
            gridContainer.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            gridContainer.widthAnchor.constraint(equalTo: view.widthAnchor, multiplier: 0.8),
            gridContainer.heightAnchor.constraint(equalTo: gridContainer.widthAnchor),
            
            // Status Label constraints
            statusLabel.topAnchor.constraint(equalTo: gridContainer.bottomAnchor, constant: 20),
            statusLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 20),
            statusLabel.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -20),
            
            // Reset Game Button constraints
            resetGameButton.topAnchor.constraint(equalTo: statusLabel.bottomAnchor, constant: 20),
            resetGameButton.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            resetGameButton.widthAnchor.constraint(equalToConstant: 160),
            resetGameButton.heightAnchor.constraint(equalToConstant: 44),
            
            // Export Button constraints
            exportButton.topAnchor.constraint(equalTo: resetGameButton.bottomAnchor, constant: 10),
            exportButton.centerXAnchor.constraint(equalTo: resetGameButton.centerXAnchor),
            exportButton.widthAnchor.constraint(equalTo: resetGameButton.widthAnchor),
            exportButton.heightAnchor.constraint(equalTo: resetGameButton.heightAnchor),
            
            // Reset Data Button constraints
            resetDataButton.topAnchor.constraint(equalTo: exportButton.bottomAnchor, constant: 10),
            resetDataButton.centerXAnchor.constraint(equalTo: resetGameButton.centerXAnchor),
            resetDataButton.widthAnchor.constraint(equalTo: resetGameButton.widthAnchor),
            resetDataButton.heightAnchor.constraint(equalTo: resetGameButton.heightAnchor)
        ])
    }
    
    // MARK: - Game Logic
    
    @objc private func buttonTapped(_ sender: UIButton) {
        let cell = sender.tag
        
        // Ignore if cell already filled or game is inactive
        if sender.title(for: .normal) != "" || !gameActive {
            return
        }
        
        // Update the tapped cell with the current symbol
        sender.setTitle(currentSymbol, for: .normal)
        
        // Record the move
        moveCount += 1
        let move = Move(user: currentPlayer, symbol: currentSymbol, cell: cell, moveNumber: moveCount)
        moves.append(move)
        
        // Check for win or draw
        if checkForWin() {
            statusLabel.text = "Player \(currentPlayer) (\(currentSymbol)) wins!"
            gameActive = false
            saveGameResult(winner: currentPlayer)
        } else if moveCount == 9 {
            statusLabel.text = "Game ended in a draw!"
            gameActive = false
            saveGameResult(winner: nil)
        } else {
            switchPlayer()
        }
    }
    
    private func switchPlayer() {
        currentPlayer = (currentPlayer == "Stalin") ? "Lenin" : "Stalin"
        currentSymbol = (currentSymbol == "X") ? "O" : "X"
        statusLabel.text = "Player: \(currentPlayer) (\(currentSymbol))"
    }
    
    private func checkForWin() -> Bool {
        // Define all winning combinations (rows, columns, diagonals)
        let winPatterns = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8], // Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8], // Columns
            [0, 4, 8], [2, 4, 6]             // Diagonals
        ]
        
        for pattern in winPatterns {
            let a = buttons[pattern[0]].title(for: .normal) ?? ""
            let b = buttons[pattern[1]].title(for: .normal) ?? ""
            let c = buttons[pattern[2]].title(for: .normal) ?? ""
            if a != "" && a == b && b == c {
                return true
            }
        }
        return false
    }
    
    @objc private func resetGame() {
        // Clear the board
        for button in buttons {
            button.setTitle("", for: .normal)
        }
        
        // Reset game state variables
        currentPlayer = "Stalin"
        currentSymbol = "X"
        gameActive = true
        moveCount = 0
        moves = []
        
        statusLabel.text = "Player: Stalin (X)"
    }
    
    // MARK: - Data Persistence
    
    private func saveGameResult(winner: String?) {
        let gameResult = GameResult(
            winner: winner,
            moves: moves,
            date: Date(),
            gameId: UUID().uuidString
        )
        
        gameResults.append(gameResult)
        saveDataToFile()
    }
    
    private func saveDataToFile() {
        guard let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
            print("Could not find documents directory")
            return
        }
        
        // Save as JSON
        let jsonURL = documentsDirectory.appendingPathComponent("tictactoe_data.json")
        do {
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            encoder.outputFormatting = .prettyPrinted
            let jsonData = try encoder.encode(gameResults)
            try jsonData.write(to: jsonURL)
            print("JSON data saved to: \(jsonURL.path)")
        } catch {
            print("Error saving JSON data: \(error)")
        }
        
        // Save as CSV
        let csvURL = documentsDirectory.appendingPathComponent("tictactoe_data.csv")
        var csvString = "GameID,Date,Winner,MoveNumber,User,Symbol,Cell\n"
        for game in gameResults {
            for move in game.moves {
                let dateFormatter = DateFormatter()
                dateFormatter.dateFormat = "yyyy-MM-dd HH:mm:ss"
                let dateString = dateFormatter.string(from: game.date)
                csvString += "\(game.gameId),\(dateString),\(game.winner ?? "Draw"),\(move.moveNumber),\(move.user),\(move.symbol),\(move.cell)\n"
            }
        }
        do {
            try csvString.write(to: csvURL, atomically: true, encoding: .utf8)
            print("CSV data saved to: \(csvURL.path)")
        } catch {
            print("Error saving CSV data: \(error)")
        }
    }
    
    private func loadSavedData() {
        guard let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
            return
        }
        
        let jsonURL = documentsDirectory.appendingPathComponent("tictactoe_data.json")
        if FileManager.default.fileExists(atPath: jsonURL.path) {
            do {
                let data = try Data(contentsOf: jsonURL)
                let decoder = JSONDecoder()
                decoder.dateDecodingStrategy = .iso8601
                gameResults = try decoder.decode([GameResult].self, from: data)
                print("Loaded \(gameResults.count) saved games")
            } catch {
                print("Error loading saved data: \(error)")
            }
        }
    }
    
    // MARK: - Data Export & Reset
    
    @objc private func exportData() {
        guard let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
            return
        }
        let jsonURL = documentsDirectory.appendingPathComponent("tictactoe_data.json")
        let csvURL = documentsDirectory.appendingPathComponent("tictactoe_data.csv")
        let activityViewController = UIActivityViewController(activityItems: [jsonURL, csvURL], applicationActivities: nil)
        if let popoverController = activityViewController.popoverPresentationController {
            popoverController.sourceView = exportButton
            popoverController.sourceRect = exportButton.bounds
        }
        present(activityViewController, animated: true)
    }
    
    @objc private func resetData() {
        // Get documents directory
        guard let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
            return
        }
        
        // Define file URLs for JSON and CSV
        let jsonURL = documentsDirectory.appendingPathComponent("tictactoe_data.json")
        let csvURL = documentsDirectory.appendingPathComponent("tictactoe_data.csv")
        let fileManager = FileManager.default
        
        // Delete JSON file if it exists
        if fileManager.fileExists(atPath: jsonURL.path) {
            do {
                try fileManager.removeItem(at: jsonURL)
                print("Deleted JSON file at: \(jsonURL.path)")
            } catch {
                print("Error deleting JSON file: \(error)")
            }
        }
        
        // Delete CSV file if it exists
        if fileManager.fileExists(atPath: csvURL.path) {
            do {
                try fileManager.removeItem(at: csvURL)
                print("Deleted CSV file at: \(csvURL.path)")
            } catch {
                print("Error deleting CSV file: \(error)")
            }
        }
        
        // Clear in-memory game results
        gameResults.removeAll()
        
        // Optionally, you can update the status label or show an alert
        statusLabel.text = "Data reset. New game ready!"
    }
}

// MARK: - AppDelegate and SceneDelegate

@main
class AppDelegate: UIResponder, UIApplicationDelegate {
    func application(_ application: UIApplication,
                     didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?) -> Bool {
        return true
    }
    
    func application(_ application: UIApplication,
                     configurationForConnecting connectingSceneSession: UISceneSession,
                     options: UIScene.ConnectionOptions) -> UISceneConfiguration {
        return UISceneConfiguration(name: "Default Configuration", sessionRole: connectingSceneSession.role)
    }
}

class SceneDelegate: UIResponder, UIWindowSceneDelegate {
    var window: UIWindow?
    
    func scene(_ scene: UIScene,
               willConnectTo session: UISceneSession,
               options connectionOptions: UIScene.ConnectionOptions) {
        guard let windowScene = (scene as? UIWindowScene) else { return }
        window = UIWindow(windowScene: windowScene)
        window?.rootViewController = ViewController()
        window?.makeKeyAndVisible()
    }
}
