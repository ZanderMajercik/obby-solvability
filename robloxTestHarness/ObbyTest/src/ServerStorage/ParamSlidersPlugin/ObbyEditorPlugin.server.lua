-- Remote Machine
local backendUrl = "http://REMOTE_IP:3000"

local PhysicsService = game:GetService("PhysicsService")
local ReplicatedStorage = game:GetService("ReplicatedStorage")
local ServerStorage = game:GetService("ServerStorage")
local HttpService = game:GetService("HttpService")
local SelectionService = game:GetService("Selection")
local Selection = game:GetService("Selection")
local Workspace = game:GetService("Workspace")
local CollectionService = game:GetService("CollectionService")
local ChangeHistoryService = game:GetService("ChangeHistoryService")
local RunService = game:GetService("RunService")
local LoadObbyFunctions = require(ServerStorage.LoadObbyFunctions)
local PlayerCloneModule = require(ServerStorage.ParamSlidersPlugin.PlayerCloneModule)

-- Plugin script
local toolbar = plugin:CreateToolbar("Obby Editor")
local button = toolbar:CreateButton(
        "Obby Editor", "Opens the dropdown menu widget", 
        "rbxassetid://4458901886"
    )

-- Create widget
local widgetInfo =
    DockWidgetPluginGuiInfo.new(Enum.InitialDockState.Float, -- Widget will be initialized in floating panel
    false, -- Widget will be initially enabled
    true, -- override the previous enabled state
    350, -- Default width
    500, -- Default height
    250, -- Minimum width (optional)
    350 -- Minimum height (optional)
    )
local widget = plugin:CreateDockWidgetPluginGuiAsync("MyPluginWidget", widgetInfo)
widget.Title = "Obby Editor"

-- Holds the state of the UI: the level options, the current parameters,
local MAX_WORLDS = 16
local currentLevelType = {}
local currentLevelJson = {}

local submissionTrajectoryOffset = 2.4

local levelTypeOptions = {}
local agentOptions = {}

local loadLevelOptions = {} -- List of obby files that can be loaded
local loadQueryOptions = {} -- List of query files (obby file + query info) that can be loaded.

local currentParameters = {}
local currentWorldIdx = 1
local trajectoryStepJob = nil
local shouldStopPolling = { false }

local simulationPaused = { false }
local simulationTimestep = 0
local searchStartTimestep = 0

local currAgentName = "Agent"

local worldTranslationOffsets = {}
-- Place worlds next to each other extending in +X.
for i = 1, MAX_WORLDS, 1 do
    worldTranslationOffsets[i] = Vector3.new((i - 1) * 50, 0, 0)
end

local selectionChangedConnection = nil
-- These are tables so they can be passed "by reference" to the checkbox generator.
local inLivePlaybackMode = { false }
local inSuccessTrajectoryMode = { false }
local inFailureTrajectoryMode = { false }
local activeTrajectoryIdx = -1

local buttonShowGoExploreBins = { true } 
local showGoExploreBins = { true } 
local showPlayerAndDynamicObjects = { true }

local numShownTrajectories = 8

-- Function to stop live playback and uncheck all related options
-- Note: Checkboxes may not be defined yet when this is called, so we check for their existence
local function stopLivePlayback()
    -- Stop the trajectory replay job if it's running
    if trajectoryStepJob then
        trajectoryStepJob:Disconnect()
        trajectoryStepJob = nil
    end
    
    -- Uncheck all playback-related options
    inLivePlaybackMode[1] = false
    inSuccessTrajectoryMode[1] = false
    inFailureTrajectoryMode[1] = false
    
    -- Update checkbox visuals (if they exist)
    if livePlaybackCheckbox then
        livePlaybackCheckbox["checkmark"].Visible = false
    end
    if showSuccessfulTrajectoriesCheckbox then
        showSuccessfulTrajectoriesCheckbox["checkmark"].Visible = false
    end
    if showFailureTrajectoriesCheckbox then
        showFailureTrajectoriesCheckbox["checkmark"].Visible = false
    end
end

local function roundCorners(button, radius)
    radius = radius or 0.1
    local cornerRadius = Instance.new("UICorner")
    cornerRadius.CornerRadius = UDim.new(radius, 0)
    cornerRadius.Parent = button
    -- Add a border
    local border = Instance.new("UIStroke")
    border.ApplyStrokeMode = Enum.ApplyStrokeMode.Border
    border.Color = Color3.fromRGB(0, 0, 0)
    border.Thickness = 1
    border.Parent = button
end

-- Create the UI elements (make it scrollable)
local frame1 = Instance.new("ScrollingFrame")
frame1.Size = UDim2.new(1, 0, 1, 0)
frame1.BackgroundColor3 = Color3.fromRGB(255, 255, 255)
frame1.ScrollBarThickness = 8
frame1.BorderSizePixel = 0
frame1.Parent = widget

-- Create content frame for scrolling
local contentFrame = Instance.new("Frame")
contentFrame.Size = UDim2.new(1, 0, 1, 0)
contentFrame.BackgroundTransparency = 1
contentFrame.Parent = frame1
contentFrame.Name = "MainContentFrame"

local frame2 = Instance.new("ScrollingFrame")
frame2.Size = UDim2.new(1, 0, 1.0 - frame1.Size.Y.Scale, 0)
frame2.BackgroundColor3 = Color3.fromRGB(255, 255, 255)
frame2.Parent = widget
frame2.Position = UDim2.new(0, 0, frame1.Size.Y.Scale, 0)

local title = Instance.new("TextLabel")
title.Text = "Current Level:"
title.Size = UDim2.new(0, 200, 0, 30)
title.Position = UDim2.new(0, 20, 0, 10)
title.BackgroundTransparency = 1
title.TextSize = 18
title.Font = Enum.Font.SourceSansBold
title.TextXAlignment = Enum.TextXAlignment.Left
title.Parent = contentFrame
title.Name = "CurrentLevelTextLabel"

function createGUIButton(text, size, position, parentFrame, connectedFunction)
    local button = Instance.new("TextButton")
    button.Text = text
    button.Size = size
    button.Position = position
    button.BackgroundColor3 = Color3.fromRGB(230, 230, 230)
    button.TextSize = 14
    button.Font = Enum.Font.SourceSans
    button.Parent = parentFrame
    roundCorners(button)
    button.MouseButton1Click:Connect(connectedFunction)
    return button
end

local saveObstacleButton = createGUIButton("Save", UDim2.new(0, 45, 0, 30), UDim2.new(0, 285, 0, 10), contentFrame,
function()
    local simPauseLatch = simulationPaused[1]
    simulationPaused[1] = true
    playOrPauseSimulation()
    setSimulationTimestep(0) -- Always save levels in the starting orientation.
    wait(0.1) -- Wait to allow the sim to correctly set timesteps on objects.

    local payload = prepareLevelParamsPayload()

    local jsonData = HttpService:JSONEncode(payload)

    -- Send the current parameters to backend
	local success, response = pcall(HttpService.PostAsync, HttpService, backendUrl .. "/saveCurrentParameters", jsonData)
    if success then
        print("Obstacle saved.")
    else
        print("Failed to save obstacle.")
    end
    simulationPaused[1] = simPauseLatch
    playOrPauseSimulation()
end)

local function cancelRequest()
    local success, response = pcall(HttpService.GetAsync, HttpService, backendUrl .. "/cancelCurrentOperation")
    if success then
        print(response)
        return success
    end
    print("Did not cancel")
    return success
end

local cancelRequestButton = createGUIButton("Pause", UDim2.new(0.5, -25, 0, 30), UDim2.new(0.5, 5, 0, 282), contentFrame,
function()
    -- Stop live playback when cancel is clicked
    stopLivePlayback()
    cancelRequest()
    -- Stop polling if it's runningd
    shouldStopPolling[1] = true
end)
cancelRequestButton.BackgroundColor3 = Color3.fromRGB(100, 100, 100)

local function createWaypointPart(waypointName, waypointInfo, sectionTagIndex)
    local part = Instance.new("Part")
    part.Name = waypointName
    part.Position = waypointInfo["position"]
    part.Size = waypointInfo["size"]
    part.CanCollide = false
    part.BrickColor = BrickColor.new("Bright yellow")
    part.Transparency = 0.5
    part.Anchored = true
    CollectionService:AddTag(part, tostring(sectionTagIndex))
    CollectionService:AddTag(part, "Waypoint")
    CollectionService:AddTag(part, "NoCollide")
    part:SetAttribute("index", waypointInfo["index"])
    part:SetAttribute("jumps", waypointInfo["jumps"])

    -- Add a label above the waypoint
    local billboardGui = Instance.new("BillboardGui")
    billboardGui.Name = "WaypointLabel"
    billboardGui.Adornee = part
    billboardGui.Size = UDim2.new(0, 80, 0, 20)
    billboardGui.StudsOffset = Vector3.new(0, part.Size.Y / 2 + 1, 0)
    billboardGui.AlwaysOnTop = true

    local textLabel = Instance.new("TextLabel")
    textLabel.Name = "Label"
    textLabel.Size = UDim2.new(1, 0, 1, 0)
    textLabel.BackgroundTransparency = 1
    textLabel.Text = tostring(waypointInfo["index"])
    textLabel.TextColor3 = Color3.new(1, 1, 0)
    textLabel.TextScaled = true
    textLabel.Font = Enum.Font.GothamBold
    textLabel.Parent = billboardGui

    billboardGui.Parent = part

    return part
end

local function createAvoidPart(avoidRegionName, avoidRegionInfo, sectionTagIndex)
    local part = Instance.new("Part")
    part.Name = avoidRegionName
    part.Position = avoidRegionInfo["position"]
    part.Size = avoidRegionInfo["size"]
    part.CanCollide = false
    part.BrickColor = BrickColor.new("Bright red")
    part.Transparency = 0.5
    part.Anchored = true
    CollectionService:AddTag(part, tostring(sectionTagIndex))
    CollectionService:AddTag(part, "Avoid")
    CollectionService:AddTag(part, "NoCollide")
    return part
end

local controlTitle = Instance.new("TextLabel")
controlTitle.Text = "Query Controls"
controlTitle.Size = UDim2.new(1, 0, 0, 30)
controlTitle.Position = UDim2.new(0, 0, 0, 90)
controlTitle.BackgroundTransparency = 1
controlTitle.TextSize = 18
controlTitle.Font = Enum.Font.SourceSansBold
controlTitle.Parent = contentFrame
controlTitle.Name = "controlTitleTextLabel"

-- Query controls
local addWaypointButton = createGUIButton("Add Waypoint", UDim2.new(0.333333, -20, 0, 30), UDim2.new(0, 20, 0, 130), contentFrame,
function()
    local obbyPartsFolder = Workspace:FindFirstChild("ObbyParts" .. tostring(currentWorldIdx))
    -- Create a new block of size (1, 1, 1) at the center of the obby
    local highestWaypointIdx = 1
    for _, child in ipairs(CollectionService:GetTagged("Waypoint")) do
        local idx = tonumber(child.Name:match("Waypoint(%d+)$"))
        if idx and idx > highestWaypointIdx then
            highestWaypointIdx = idx
        end
    end
    local waypointInfo = {}
    waypointInfo["position"] = Vector3.new(0, 1, 0)
    waypointInfo["size"] = Vector3.new(1, 1, 1)
    waypointInfo["index"] = highestWaypointIdx
    waypointInfo["jumps"] = -1
    local newWaypoint = createWaypointPart(
        "Waypoint" .. tostring(highestWaypointIdx + 1), 
        waypointInfo, 
        currentWorldIdx
    )
    newWaypoint.Parent = obbyPartsFolder
    LoadObbyFunctions.adjustWaypointLabels()
end)

local addAvoidRegionButton = createGUIButton("Add Avoid Region", UDim2.new(0.333333, -20, 0, 30), UDim2.new(0.333333, 10, 0, 130), contentFrame,
function()
    local obbyPartsFolder = Workspace:FindFirstChild("ObbyParts" .. tostring(currentWorldIdx))
    -- Create a new block of size (1, 1, 1) at the center of the obby
    local highestAvoidRegionIdx = 0
    for _, child in ipairs(CollectionService:GetTagged("Avoid")) do
        local idx = tonumber(child.Name:match("AvoidRegion(%d+)$"))
        if idx and idx > highestAvoidRegionIdx then
            highestAvoidRegionIdx = idx
        end
    end
    local avoidRegionInfo = {}
    avoidRegionInfo["position"] = Vector3.new(0, 1, 0)
    avoidRegionInfo["size"] = Vector3.new(1, 1, 1)
    local newAvoidRegion = createAvoidPart(
        "AvoidRegion" .. tostring(highestAvoidRegionIdx + 1), 
        avoidRegionInfo,
        currentWorldIdx
    )
    newAvoidRegion.Parent = obbyPartsFolder
end)

local obstacleDropdownContainer = Instance.new("Frame")
obstacleDropdownContainer.Size = UDim2.new(1, -40, 0, 0) -- Will expand as we add options
obstacleDropdownContainer.Position = UDim2.new(0, 20, 0, 40)
obstacleDropdownContainer.BackgroundColor3 = Color3.fromRGB(240, 240, 240)
obstacleDropdownContainer.BorderSizePixel = 1
obstacleDropdownContainer.Visible = false
obstacleDropdownContainer.ZIndex = 20
obstacleDropdownContainer.Parent = contentFrame
local obstacleDropdown = Instance.new("TextLabel")
obstacleDropdown.BackgroundTransparency = 1
obstacleDropdown.Text = currentLevelType[currentWorldIdx] or "None"
obstacleDropdown.TextSize = 14
obstacleDropdown.Font = Enum.Font.SourceSans
obstacleDropdown.TextColor3 = Color3.fromRGB(0, 0, 0)
obstacleDropdown.TextXAlignment = Enum.TextXAlignment.Left
obstacleDropdown.ZIndex = 20
obstacleDropdown.Parent = contentFrame
obstacleDropdown.Name = "obstacleDropdownTextLabel"

-- Agent dropdown UI elements
local agentDropdownContainer = Instance.new("Frame")
agentDropdownContainer.Size = UDim2.new(1, -40, 0, 0) -- Will expand as we add options
agentDropdownContainer.Position = UDim2.new(0, 20, 0, 202)
agentDropdownContainer.BackgroundColor3 = Color3.fromRGB(240, 240, 240)
agentDropdownContainer.BorderSizePixel = 1
agentDropdownContainer.Visible = false
agentDropdownContainer.ZIndex = 20
agentDropdownContainer.Parent = contentFrame
local agentDropdown = Instance.new("TextLabel")
agentDropdown.BackgroundTransparency = 1
agentDropdown.Text = currAgentName or "Agent"
agentDropdown.TextSize = 14
agentDropdown.Font = Enum.Font.SourceSans
agentDropdown.TextColor3 = Color3.fromRGB(0, 0, 0)
agentDropdown.TextXAlignment = Enum.TextXAlignment.Left
agentDropdown.ZIndex = 20
agentDropdown.Parent = contentFrame
agentDropdown.Name = "agentDropdownTextLabel"

local cameraOptionsTitle = nil
local viewCameraButton = nil
local topCameraButton = nil
local sideCameraButton = nil

local loadObstacleButton = createGUIButton("Load", UDim2.new(0, 45, 0, 30), UDim2.new(0, 230, 0, 10), contentFrame,
function()
    local showMenus = not obstacleDropdownContainer.Visible
    if showMenus then
        loadLevelOptions = fetchLoadLevelOptions() or {}
        levelTypeOptions = fetchLevelOptions() or {}
        agentOptions = fetchAgentOptions() or {}
        updateDropdownOptions()
    end

    obstacleDropdownContainer.Visible = showMenus
end)

local selectAgentButton = createGUIButton("Agent", UDim2.new(0, 50, 0, 30), UDim2.new(0, 280, 0, 202), contentFrame,
function()
    local showMenus = not agentDropdownContainer.Visible
    if showMenus then
        agentOptions = fetchAgentOptions() or {}
        updateAgentDropdownOptions()
    end

    agentDropdownContainer.Visible = showMenus
end)

-- Create agent title label
local agentTitle = Instance.new("TextLabel")
agentTitle.Text = "Agent:"
agentTitle.Size = UDim2.new(0, 70, 0, 30)
agentTitle.Position = UDim2.new(0, 20, 0, 202)
agentTitle.BackgroundTransparency = 1
agentTitle.TextSize = 14
agentTitle.Font = Enum.Font.SourceSans
agentTitle.TextXAlignment = Enum.TextXAlignment.Left
agentTitle.Parent = contentFrame
agentTitle.Name = "agentTitleTextLabel"

local function layoutLevelHeader()
    local frameWidth = contentFrame.AbsoluteSize.X
    if frameWidth <= 0 then
        return
    end

    local leftPadding = 20
    local interItemGap = 10
    local rightPadding = 20

    local saveWidth = math.clamp(math.floor(frameWidth * 0.12), 45, 80)
    local loadWidth = math.clamp(math.floor(frameWidth * 0.12), 45, 80)
    local titleWidth = math.clamp(math.floor(frameWidth * 0.24), 90, 180)
    local saveX = frameWidth - rightPadding - saveWidth
    local loadX = saveX - interItemGap - loadWidth
    local levelWidth = loadX - (leftPadding + titleWidth + interItemGap)
    levelWidth = math.max(levelWidth, 80)

    title.Size = UDim2.new(0, titleWidth, 0, 30)
    title.Position = UDim2.new(0, leftPadding, 0, 10)

    obstacleDropdown.Size = UDim2.new(0, levelWidth, 0, 30)
    obstacleDropdown.Position = UDim2.new(0, leftPadding + titleWidth + interItemGap, 0, 10)

    loadObstacleButton.Size = UDim2.new(0, loadWidth, 0, 30)
    loadObstacleButton.Position = UDim2.new(0, loadX, 0, 10)

    saveObstacleButton.Size = UDim2.new(0, saveWidth, 0, 30)
    saveObstacleButton.Position = UDim2.new(0, saveX, 0, 10)

    obstacleDropdownContainer.Position = UDim2.new(0, 20, 0, 45)
    obstacleDropdownContainer.Size = UDim2.new(1, -40, 0, 0)

    -- Agent dropdown positioning
    local agentButtonWidth = math.clamp(math.floor(frameWidth * 0.12), 45, 80)
    local agentTitleWidth = math.clamp(math.floor(frameWidth * 0.15), 70, 120)
    local agentRowY = 202 -- Title row + obstacle row + gap
    local agentWidth = frameWidth - leftPadding - agentTitleWidth - interItemGap - agentButtonWidth - rightPadding

    agentTitle.Size = UDim2.new(0, agentTitleWidth, 0, 30)
    agentTitle.Position = UDim2.new(0, leftPadding, 0, agentRowY)

    agentDropdown.Size = UDim2.new(0, agentWidth, 0, 30)
    agentDropdown.Position = UDim2.new(0, leftPadding + agentTitleWidth + interItemGap, 0, agentRowY)

    selectAgentButton.Size = UDim2.new(0, agentButtonWidth, 0, 30)
    selectAgentButton.Position = UDim2.new(0, frameWidth - rightPadding - agentButtonWidth, 0, agentRowY)

    agentDropdownContainer.Position = UDim2.new(0, 20, 0, agentRowY + 35)
    agentDropdownContainer.Size = UDim2.new(1, -40, 0, 0)

    local cameraTitleWidth = math.clamp(math.floor(frameWidth * 0.18), 90, 160)
    local cameraButtonWidth = math.floor((frameWidth - leftPadding - cameraTitleWidth - (interItemGap * 3) - rightPadding) / 3)
    cameraButtonWidth = math.max(cameraButtonWidth, 60)
    local cameraRowY = 10 + 30 + 10

    cameraOptionsTitle.Size = UDim2.new(0, cameraTitleWidth, 0, 30)
    cameraOptionsTitle.Position = UDim2.new(0, leftPadding, 0, cameraRowY)

    if viewCameraButton then
        viewCameraButton.Size = UDim2.new(0, cameraButtonWidth, 0, 30)
        viewCameraButton.Position = UDim2.new(0, leftPadding + cameraTitleWidth + interItemGap, 0, cameraRowY)
    end

    if topCameraButton then
        topCameraButton.Size = UDim2.new(0, cameraButtonWidth, 0, 30)
        topCameraButton.Position = UDim2.new(0, leftPadding + cameraTitleWidth + interItemGap + cameraButtonWidth + interItemGap, 0, cameraRowY)
    end

    if sideCameraButton then
        sideCameraButton.Size = UDim2.new(0, cameraButtonWidth, 0, 30)
        sideCameraButton.Position = UDim2.new(0, leftPadding + cameraTitleWidth + interItemGap + (cameraButtonWidth + interItemGap) * 2, 0, cameraRowY)
    end
end

contentFrame:GetPropertyChangedSignal("AbsoluteSize"):Connect(layoutLevelHeader)
task.defer(layoutLevelHeader)

cameraOptionsTitle = Instance.new("TextLabel")
cameraOptionsTitle.Text = "Cameras:"
cameraOptionsTitle.Size = UDim2.new(0, 100, 0, 30)
cameraOptionsTitle.Position = UDim2.new(0, 20, 0, 90)
cameraOptionsTitle.BackgroundTransparency = 1
cameraOptionsTitle.TextSize = 18
cameraOptionsTitle.Font = Enum.Font.SourceSansBold
cameraOptionsTitle.TextXAlignment = Enum.TextXAlignment.Left
cameraOptionsTitle.Parent = contentFrame
cameraOptionsTitle.Name = "cameraOptionsTitleTextLabel"

-- Create a scrolling frame for the parameter sliders
local parametersScrollFrame = Instance.new("ScrollingFrame")
parametersScrollFrame.Size = UDim2.new(0,0,0,0)
parametersScrollFrame.Position = UDim2.new(0.05, 0, 0, 200)
parametersScrollFrame.BackgroundColor3 = Color3.fromRGB(245, 245, 245)
parametersScrollFrame.BorderSizePixel = 1
parametersScrollFrame.ScrollBarThickness = 0
parametersScrollFrame.Visible = false
parametersScrollFrame.Parent = contentFrame

-- Create a frame for the checkboxes
local analysisFrame = Instance.new("Frame")
analysisFrame.Size = UDim2.new(1, -40, 0, 120)
-- Add a title above the analysis frame and shift the frame down by the title height (30px)
local analysisTitle = Instance.new("TextLabel")
analysisTitle.Text = "Visualization Tool"
analysisTitle.Size = UDim2.new(1, 0, 0, 30)
analysisTitle.Position = UDim2.new(0, 0, 0, 0)
analysisTitle.BackgroundTransparency = 1
analysisTitle.TextSize = 16
analysisTitle.Font = Enum.Font.SourceSansBold
analysisTitle.TextXAlignment = Enum.TextXAlignment.Center
analysisTitle.Parent = analysisFrame
analysisTitle.Name = "analysisTitleTextLabel"

-- Position the analysis frame (restore original vertical offset)
analysisFrame.Position = UDim2.new(0, 20, parametersScrollFrame.Size.Y.Scale, parametersScrollFrame.Position.Y.Offset + 123)
analysisFrame.BackgroundColor3 = Color3.fromRGB(245, 245, 245)
analysisFrame.Parent = contentFrame
analysisFrame.Visible = false

-- Create a frame for the block editor
local blockEditorFrame = Instance.new("Frame")
blockEditorFrame.Size = UDim2.new(1, -40, 0, 300)
blockEditorFrame.Position = UDim2.new(0, 20, parametersScrollFrame.Size.Y.Scale + analysisFrame.Size.Y.Scale, analysisFrame.Position.Y.Offset + analysisFrame.Size.Y.Offset + 10)
blockEditorFrame.BackgroundColor3 = Color3.fromRGB(240, 248, 255)
blockEditorFrame.BorderSizePixel = 1
blockEditorFrame.BorderColor3 = Color3.fromRGB(100, 100, 100)
blockEditorFrame.Parent = contentFrame
blockEditorFrame.Visible = false

-- Add title for block editor frame
local blockEditorTitle = Instance.new("TextLabel")
blockEditorTitle.Text = "Block Editor"
blockEditorTitle.Size = UDim2.new(1, 0, 0, 30)
blockEditorTitle.Position = UDim2.new(0, 0, 0, 0)
blockEditorTitle.BackgroundTransparency = 1
blockEditorTitle.TextSize = 16
blockEditorTitle.Font = Enum.Font.SourceSansBold
blockEditorTitle.TextXAlignment = Enum.TextXAlignment.Center
blockEditorTitle.Parent = blockEditorFrame
blockEditorTitle.Name = "blockEditorTitleTextLabel"

-- Helper function to create a label
local function createLabel(text, size, position, parent)
    local label = Instance.new("TextLabel")
    label.Text = text
    label.Size = size
    label.Position = position
    label.BackgroundTransparency = 1
    label.TextSize = 16
    label.Font = Enum.Font.SourceSans
    label.TextXAlignment = Enum.TextXAlignment.Left
    label.ZIndex = 12 -- Higher than click button (10) but lower than inputs (15)
    label.Parent = parent
    return label
end

-- Helper function to create a TextBox
local function createTextBox(size, position, placeholderText, parent, defaultValue)
    local textBox = Instance.new("TextBox")
    textBox.Size = size
    textBox.Position = position
    textBox.BackgroundColor3 = Color3.fromRGB(255, 255, 255)
    textBox.BorderSizePixel = 1
    textBox.Text = defaultValue or ""
    textBox.PlaceholderText = placeholderText
    textBox.TextSize = 14
    textBox.Font = Enum.Font.SourceSans
    textBox.TextXAlignment = Enum.TextXAlignment.Left
    textBox.ClearTextOnFocus = false
    textBox.ZIndex = 15 -- Higher than click button (10) to allow interaction
    textBox.Parent = parent
    roundCorners(textBox, 0.05)
    return textBox
end

-- Helper function to create a dropdown
local function createDropdown(size, position, options, parent, defaultIndex, onChangeCallback)
    local dropdown = Instance.new("TextButton")
    dropdown.Size = size
    dropdown.Position = position
    dropdown.BackgroundColor3 = Color3.fromRGB(255, 255, 255)
    dropdown.BorderSizePixel = 1
    dropdown.Text = options[defaultIndex or 1]
    dropdown.TextSize = 14
    dropdown.Font = Enum.Font.SourceSans
    dropdown.TextColor3 = Color3.fromRGB(0, 0, 0)
    dropdown.ZIndex = 15 -- Higher than click button (10) to allow interaction
    dropdown.Parent = parent
    roundCorners(dropdown, 0.05)
    
    -- Store current index
    local currentIndex = defaultIndex or 1
    dropdown.MouseButton1Click:Connect(function()
        currentIndex = (currentIndex % #options) + 1
        dropdown.Text = options[currentIndex]
        -- Call the callback if provided
        if onChangeCallback then
            onChangeCallback()
        end
    end)
    
    return dropdown
end

local maxTrajectorySteps = 3000
local maxTrajStepsLabel = createLabel(
    "Max Steps",
    UDim2.new(0, 75, 0, 20),   -- width/height of label
    UDim2.new(0, 20, 0, 170),   -- position within contentFrame (X scale, X offset, Y scale, Y offset)
    contentFrame
)
maxTrajStepsLabel.Font = Enum.Font.SourceSansBold
maxTrajStepsLabel.ZIndex = 5

local maxTrajStepsBox = createTextBox(
    UDim2.new(0.5, -100, 0, 22), -- textbox size
    UDim2.new(0, 95, 0, 170), -- textbox position
    "0",                         -- placeholder
    contentFrame,
    tostring(maxTrajectorySteps) -- default text
)
maxTrajStepsBox.ZIndex = 5

function updateTextBox(textbox, value)
    textbox.Text = value
end
maxTrajStepsBox.FocusLost:Connect(function()
    local v = tonumber(maxTrajStepsBox.Text)
    if not v then
        v = 0
    end
    maxTrajectorySteps = v
    -- normalize text to the parsed value
    updateTextBox(maxTrajStepsBox, tostring(maxTrajectorySteps))
end)

local maxTrajectoryJumps = 3000
local maxTrajJumpsLabel = createLabel(
    "Max Jumps",
    UDim2.new(0, 75, 0, 20),   -- width/height of label
    UDim2.new(0.5, 5, 0, 170), -- position within contentFrame (X scale, X offset, Y scale, Y offset)
    contentFrame
)
maxTrajJumpsLabel.Font = Enum.Font.SourceSansBold
maxTrajJumpsLabel.ZIndex = 5

local maxTrajJumpsBox = createTextBox(
    UDim2.new(0.5, -100, 0, 22), -- textbox size
    UDim2.new(0.5, 80, 0, 170), -- textbox position
    "0",                         -- placeholder
    contentFrame,
    tostring(maxTrajectoryJumps) -- default text
)
maxTrajJumpsBox.ZIndex = 5

maxTrajJumpsBox.FocusLost:Connect(function()
    local v = tonumber(maxTrajJumpsBox.Text)
    if not v then
        v = 0
    end
    maxTrajectoryJumps = v
    -- normalize text to the parsed value
    updateTextBox(maxTrajJumpsBox, tostring(maxTrajectoryJumps))
end)

local simTimeStepLabel = createLabel(
    "Sim Timestep",
    UDim2.new(0, 80, 0, 20),   -- width/height of label
    UDim2.new(0, 20, 0.35, 40),  -- position within contentFrame (X scale, X offset, Y scale, Y offset)
    analysisFrame
)
simTimeStepLabel.ZIndex = 5

local simTimeStepBox = createTextBox(
    UDim2.new(0, 60, 0, 22),    -- textbox size
    UDim2.new(0, 100, 0.35, 40), -- textbox position
    "0",                         -- placeholder
    analysisFrame,
    tostring(simulationTimestep) -- default text
)
simTimeStepBox.ZIndex = 5

simTimeStepBox.FocusLost:Connect(function()
    local v = tonumber(simTimeStepBox.Text)
    if not v then
        v = 0
    end
    setSimulationTimestep(v)
    simulationTimestep = v
    -- normalize text to the parsed value
    updateTextBox(simTimeStepBox, tostring(simulationTimestep))
end)

RunService.Heartbeat:Connect(function()
    if not simulationPaused[1] and simTimeStepBox then
        simTimeStepBox.Text = tostring(simulationTimestep)
    end
end)

-- Behavior frame

local behaviorFrame = Instance.new("Frame")
behaviorFrame.Size = UDim2.new(1/3, -80/3, 0, 250)
behaviorFrame.Position = UDim2.new(0, 20, 0, 35)
behaviorFrame.BackgroundColor3 = Color3.fromRGB(240, 240, 240) -- Lighter gray (no behavior)
behaviorFrame.BorderSizePixel = 1
behaviorFrame.Parent = blockEditorFrame

-- Create PersistentRotate behavior frame (Column 2)
local rotateFrame = Instance.new("Frame")
rotateFrame.Size = UDim2.new(1/3, -80/3, 0, 250)
rotateFrame.Position = UDim2.new(1/3, 40/3, 0, 35)
rotateFrame.BackgroundColor3 = Color3.fromRGB(240, 240, 240) -- Lighter gray (no behavior)
rotateFrame.BorderSizePixel = 1
rotateFrame.Parent = blockEditorFrame

-- Declare these variables first
local rotateAxisDropdown = nil
local rotateAngularBox = nil
local translateTimeBox = nil
local translateStartBox = nil
local translateEndBox = nil
local currentSelectedBlock = nil
local selectedBlockName = nil -- Track the name of the selected block

function playOrPauseSimulation()
    for _, part in pairs(CollectionService:GetTagged("obby")) do
        local simulate = part:GetAttribute("Simulate")
        if simulate ~= nil then
            part:SetAttribute("Simulate", not simulationPaused[1])
        end
    end
end

function setSimulationTimestep(t)
    for _, part in pairs(CollectionService:GetTagged("obby")) do
        local timestep = part:GetAttribute("Timestep")
        if timestep ~= nil then
            part:SetAttribute("Timestep", t)
        end
    end
end

-- Function to reselect a block by name after level reload
local function reselectBlockByName()
    -- Wait a frame for the level to finish loading
    RunService.Heartbeat:Wait()
    
    if selectedBlockName == nil then
        return
    end

    -- Search for the block by name in the current world
    local obbyFolder = Workspace:FindFirstChild("ObbyParts" .. tostring(currentWorldIdx))
    if not obbyFolder then
        return
    end
    
    -- Find the block by name
    local targetBlock = obbyFolder:FindFirstChild(selectedBlockName)
    if targetBlock and targetBlock:IsA("BasePart") then
        -- Reselect the block
        SelectionService:Set({targetBlock})
    end
end

-- Function to update behavior parameters when fields change
local function updateBehaviorParameters()
    if not currentSelectedBlock then
        return
    end
    
    -- Update Rotate parameters if behavior exists
    if currentSelectedBlock:HasTag("PersistentRotate") then
        -- Convert axis dropdown text to attribute format
        local axisMap = {["X"] = "right", ["Y"] = "up", ["Z"] = "forward"}
        local axisValue = axisMap[rotateAxisDropdown.Text] or "right"
        currentSelectedBlock:SetAttribute("axis", axisValue)
        currentSelectedBlock:SetAttribute("angular", tonumber(rotateAngularBox.Text) or 2.36)
    end
    
    -- Update Translate parameters if behavior exists
    if currentSelectedBlock:HasTag("PersistentTranslate") then
        currentSelectedBlock:SetAttribute("time", tonumber(translateTimeBox.Text) or 1.0)
        do
            local x, y, z = translateStartBox.Text:match("^%s*([-%d%.]+)%s*,%s*([-%d%.]+)%s*,%s*([-%d%.]+)%s*$")
            if x and y and z then
                local vec = Vector3.new(tonumber(x), tonumber(y), tonumber(z))
                currentSelectedBlock:SetAttribute("startPosition", HttpService:JSONEncode({vec.X, vec.Y, vec.Z}))
            else
                -- Fallback for invalid input
                currentSelectedBlock:SetAttribute("startPosition", HttpService:JSONEncode({0, 0, 0}))
            end
        end
        currentSelectedBlock:SetAttribute("endPosition", "[" .. translateEndBox.Text .. "]")
    end
    
    -- Resend obby JSON to backend and reload level, then reselect
    sendParametersAndUpdateLevel()
    reselectBlockByName()
end

local rotateTitle = createLabel("Rotate", UDim2.new(0.9, 0, 0, 20), UDim2.new(0.05, 0, 0, 5), rotateFrame)
createLabel("Axis:", UDim2.new(0.9, 0, 0, 20), UDim2.new(0.05, 0, 0, 30), rotateFrame)
rotateAxisDropdown = createDropdown(UDim2.new(0.85, 0, 0, 25), UDim2.new(0.075, 0, 0, 55), {"X", "Y", "Z"}, rotateFrame, 1, updateBehaviorParameters)
createLabel("Angular:", UDim2.new(0.9, 0, 0, 20), UDim2.new(0.05, 0, 0, 90), rotateFrame)
rotateAngularBox = createTextBox(UDim2.new(0.85, 0, 0, 25), UDim2.new(0.075, 0, 0, 115), "0.0", rotateFrame, "2.36")

-- Create PersistentTranslate behavior frame (Column 3)
local translateFrame = Instance.new("Frame")
translateFrame.Size = UDim2.new(1/3, -80/3, 0, 250)
translateFrame.Position = UDim2.new(2/3, 20/3, 0, 35)
translateFrame.BackgroundColor3 = Color3.fromRGB(240, 240, 240) -- Lighter gray (no behavior)
translateFrame.BorderSizePixel = 1
translateFrame.Parent = blockEditorFrame

local translateTitle = createLabel("Translate", UDim2.new(0.9, 0, 0, 20), UDim2.new(0.05, 0, 0, 5), translateFrame)

-- Time field
createLabel("Time:", UDim2.new(0.9, 0, 0, 18), UDim2.new(0.05, 0, 0, 30), translateFrame)
translateTimeBox = createTextBox(UDim2.new(0.85, 0, 0, 22), UDim2.new(0.075, 0, 0, 50), "0.0", translateFrame, "1.0")

-- Start position field
createLabel("Start (x,y,z):", UDim2.new(0.9, 0, 0, 18), UDim2.new(0.05, 0, 0, 78), translateFrame)
translateStartBox = createTextBox(UDim2.new(0.85, 0, 0, 22), UDim2.new(0.075, 0, 0, 100), "0,0,0", translateFrame, "0,0,0")

-- End position field
createLabel("End (x,y,z):", UDim2.new(0.9, 0, 0, 18), UDim2.new(0.05, 0, 0, 130), translateFrame)
translateEndBox = createTextBox(UDim2.new(0.85, 0, 0, 22), UDim2.new(0.075, 0, 0, 152), "0,0,0", translateFrame, "10,0,0")

-- Connect field change events
rotateAngularBox.FocusLost:Connect(updateBehaviorParameters)
translateTimeBox.FocusLost:Connect(updateBehaviorParameters)
translateStartBox.FocusLost:Connect(updateBehaviorParameters)
translateEndBox.FocusLost:Connect(updateBehaviorParameters)

-- Helper function to create an Add/Remove button
local function createAddRemoveButton(parent, yPos, width)
    local button = Instance.new("TextButton")
    button.Size = UDim2.new(width, 0, 0, 25)
    button.Position = UDim2.new((1 - width) / 2, 0, 0, yPos)
    button.BackgroundColor3 = Color3.fromRGB(50, 120, 220)
    button.TextColor3 = Color3.fromRGB(255, 255, 255)
    button.TextSize = 14
    button.Font = Enum.Font.SourceSansBold
    button.Text = "Add"
    button.Parent = parent
    roundCorners(button, 0.05)
    return button
end

-- Add block button
local addBlockButton = createAddRemoveButton(behaviorFrame, 20, 0.9)
addBlockButton.Text = "Add Block"
addBlockButton.MouseButton1Click:Connect(function()
    local obbyFolder = getFolderElseCreate(Workspace, "ObbyParts" .. tostring(currentWorldIdx))

    local blockIndex = 1
    local blockName = "Block" .. tostring(blockIndex)
    while obbyFolder:FindFirstChild(blockName) do
        blockIndex += 1
        blockName = "Block" .. tostring(blockIndex)
    end

    local newBlock = Instance.new("Part")
    newBlock.Name = blockName
    newBlock.Size = Vector3.new(4, 1, 4)
    newBlock.Position = currentSelectedBlock and (currentSelectedBlock.Position + Vector3.new(6, 0, 0)) or Vector3.new(0, 5, 0)
    newBlock.Orientation = Vector3.new(0, 0, 0)
    newBlock.Anchored = true
    newBlock.CanCollide = true
    newBlock.CanTouch = true
    newBlock.CanQuery = true
    newBlock.Color = Color3.fromRGB(98, 97, 99)
    newBlock.Parent = obbyFolder

    CollectionService:AddTag(newBlock, tostring(currentWorldIdx))
    CollectionService:AddTag(newBlock, "obby")

    currentSelectedBlock = newBlock
    selectedBlockName = newBlock.Name
    SelectionService:Set({newBlock})
end)

-- Add Remove button to Lava frame
local lavaAddRemoveButton = createAddRemoveButton(behaviorFrame, 50, 0.9)
lavaAddRemoveButton.Text = "Set Lava"

-- Add Set Spawn button to Spawn frame
local setSpawnButton = createAddRemoveButton(behaviorFrame, 80, 0.9)
setSpawnButton.Text = "Set Spawn"
setSpawnButton.BackgroundColor3 = Color3.fromRGB(50, 180, 50) -- Green color

-- Add Set Goal button to Goal frame
local setGoalButton = createAddRemoveButton(behaviorFrame, 110, 0.9)
setGoalButton.Text = "Set Goal"
setGoalButton.BackgroundColor3 = Color3.fromRGB(45, 108, 220) -- Blue color

-- Add Remove button to Rotate frame  
local rotateAddRemoveButton = createAddRemoveButton(rotateFrame, 145, 0.7)
rotateAddRemoveButton.Text = "Add"

-- Add Remove button to Translate frame
local translateAddRemoveButton = createAddRemoveButton(translateFrame, 180, 0.7)
translateAddRemoveButton.Text = "Add"

-- Increase block editor frame size to acc  ommodate all elements
blockEditorFrame.Size = UDim2.new(1, -40, 0, 300)

-- Set the content frame canvas size for scrolling
contentFrame.Size = UDim2.new(1, 0, 0, 1500)
frame1.CanvasSize = UDim2.new(0, 0, 0, 1500)

-- Connect click buttons to toggle behaviors (will be defined later)
-- These connections will be set up after toggle functions are defined

-- Function to load block behaviors into BlockEditor
local function loadBlockBehaviors(blockPart)
    if not blockPart then
        return
    end
    
    -- Get Lava behavior and update Lava frame
    if blockPart:HasTag("Damage") then
        lavaAddRemoveButton.Text = "Unset Lava"
        lavaAddRemoveButton.BackgroundColor3 = Color3.fromRGB(220, 60, 60) -- Red for remove
    else
        lavaAddRemoveButton.Text = "Set Lava"
        lavaAddRemoveButton.BackgroundColor3 = Color3.fromRGB(50, 120, 220) -- Blue for add
    end

    -- Get Rotate behavior and update Rotate frame
    if blockPart:HasTag("PersistentRotate") then
        rotateFrame.BackgroundColor3 = Color3.fromRGB(180, 180, 180) -- Darker gray
        if rotateTitle then
            rotateTitle.Font = Enum.Font.SourceSansBold
        end
        rotateAddRemoveButton.Text = "Remove"
        rotateAddRemoveButton.BackgroundColor3 = Color3.fromRGB(220, 60, 60) -- Red for remove
        -- Load axis and angular if available
        if blockPart:GetAttribute("axis") then
            if blockPart:GetAttribute("axis") == "up" then
                rotateAxisDropdown.Text = "Y"
            elseif blockPart:GetAttribute("axis") == "forward" then
                rotateAxisDropdown.Text = "Z"
            elseif blockPart:GetAttribute("axis") == "right" then
                rotateAxisDropdown.Text = "X"
            end
        end
        if blockPart:GetAttribute("angular") then
            local angularValue = blockPart:GetAttribute("angular")
            if angularValue then
                rotateAngularBox.Text = string.format("%.2f", tonumber(angularValue))
            end
        end
    else
        rotateFrame.BackgroundColor3 = Color3.fromRGB(240, 240, 240) -- Lighter gray
        if rotateTitle then
            rotateTitle.Font = Enum.Font.SourceSans
        end
        rotateAddRemoveButton.Text = "Add"
        rotateAddRemoveButton.BackgroundColor3 = Color3.fromRGB(50, 120, 220) -- Blue for add
        rotateAxisDropdown.Text = "X"
        rotateAngularBox.Text = "2.36"
    end

    -- Get Translate behavior and update Translate frame
    if blockPart:HasTag("PersistentTranslate") then
        translateFrame.BackgroundColor3 = Color3.fromRGB(180, 180, 180) -- Darker gray
        if translateTitle then
            translateTitle.Font = Enum.Font.SourceSansBold
        end
        translateAddRemoveButton.Text = "Remove"
        translateAddRemoveButton.BackgroundColor3 = Color3.fromRGB(220, 60, 60) -- Red for remove
        -- Load start, end, time if available
        if blockPart:GetAttribute("startPosition") then
            startPosStr = tostring(blockPart:GetAttribute("startPosition"))
            -- Remove brackets if they exist
            startPosStr = startPosStr:gsub("[%[%]]", "")
            translateStartBox.Text = startPosStr
        end
        if blockPart:GetAttribute("endPosition") then
            endPosStr = tostring(blockPart:GetAttribute("endPosition"))
            -- Remove brackets if they exist
            endPosStr = endPosStr:gsub("[%[%]]", "")
            translateEndBox.Text = endPosStr
        end
        if blockPart:GetAttribute("time") then
            translateTimeBox.Text = tostring(blockPart:GetAttribute("time"))
        end
    else
        translateFrame.BackgroundColor3 = Color3.fromRGB(240, 240, 240) -- Lighter gray
        if translateTitle then
            translateTitle.Font = Enum.Font.SourceSans
        end
        translateAddRemoveButton.Text = "Add"
        translateAddRemoveButton.BackgroundColor3 = Color3.fromRGB(50, 120, 220) -- Blue for add
        translateStartBox.Text = "0,0,0"
        translateEndBox.Text = "10,0,0"
        translateTimeBox.Text = "1.0"
    end

end

-- Function to toggle Lava behavior on/off
local function toggleLavaBehavior()
    if not currentSelectedBlock then
        return
    end
    
    local hasBehavior = currentSelectedBlock:HasTag("Damage")
    
    if hasBehavior then
        -- Remove the Damage tag from the block
        print("Removing Lava behavior from block")
        currentSelectedBlock:RemoveTag("Damage")
        currentSelectedBlock:SetAttribute("damage", nil)
    else
        -- Add the Damage tag to the block
        print("Adding Lava behavior to block")
        currentSelectedBlock:AddTag("Damage")
        currentSelectedBlock:SetAttribute("damage", 100)
    end
    
    -- Resend obby JSON to backend and reload level to update block appearances
    sendParametersAndUpdateLevel()
    
    -- Reselect the block after level reload
    reselectBlockByName()
end

-- Function to set Spawn behavior on the current block
function setSpawnBehavior()
    if not currentSelectedBlock then
        return
    end
    
    -- Remove Spawn tag from all other blocks in the Obby
    local allSpawnBlocks = CollectionService:GetTagged("Spawn")
    for _, block in ipairs(allSpawnBlocks) do
        if block ~= currentSelectedBlock then
            print("Removing Spawn behavior from block:", block.Name)
            CollectionService:RemoveTag(block, "Spawn")
        end
    end
    
    -- Add Spawn tag to the current block
    if not CollectionService:HasTag(currentSelectedBlock, "Spawn") then
        print("Adding Spawn behavior to block:", currentSelectedBlock.Name)
        CollectionService:AddTag(currentSelectedBlock, "Spawn")
    end
    
    -- Resend obby JSON to backend and reload level to update block appearances
    sendParametersAndUpdateLevel()
    
    -- Reselect the block after level reload
    reselectBlockByName()
end

function setGoalBehavior()
    if not currentSelectedBlock then
        return
    end
    
    -- Remove Spawn tag from all other blocks in the Obby
    local allSpawnBlocks = CollectionService:GetTagged("Checkpoint")
    for _, block in ipairs(allSpawnBlocks) do
        if block ~= currentSelectedBlock then
            CollectionService:RemoveTag(block, "Goal")
            CollectionService:RemoveTag(block, "Checkpoint")
            CollectionService:RemoveTag(block, "2")
        end
    end
    
    -- Add Spawn tag to the current block
    if not CollectionService:HasTag(currentSelectedBlock, "Goal") then
        CollectionService:AddTag(currentSelectedBlock, "Goal")
        CollectionService:AddTag(currentSelectedBlock, "Checkpoint")
        CollectionService:AddTag(currentSelectedBlock, "2")
    end
    
    -- Resend obby JSON to backend and reload level to update block appearances
    sendParametersAndUpdateLevel()
    
    -- Reselect the block after level reload
    reselectBlockByName()
end

-- Function to toggle Rotate behavior on/off
local function toggleRotateBehavior()
    if not currentSelectedBlock then
        return
    end
    
    local hasBehavior = currentSelectedBlock:HasTag("PersistentRotate")
    
    if hasBehavior then
        -- Remove the PersistentRotate tag and any associated attributes/scripts
        print("Removing Rotate behavior from block")
        currentSelectedBlock:RemoveTag("PersistentRotate")
        currentSelectedBlock:SetAttribute("axis", nil)
        currentSelectedBlock:SetAttribute("angular", nil)
        currentSelectedBlock:SetAttribute("loop", nil)
        currentSelectedBlock:SetAttribute("rotateType", nil)
        -- Remove the Rotate script if it exists
        local rotateScript = currentSelectedBlock:FindFirstChild("Rotate")
        if rotateScript then
            rotateScript:Destroy()
        end
    else
        -- Add the PersistentRotate tag and any associated attributes/scripts
        print("Adding Rotate behavior to block")
        currentSelectedBlock:AddTag("PersistentRotate")
        -- Normalize axis to the loader's expected values
        local axisMap = {["X"] = "right", ["Y"] = "up", ["Z"] = "forward"}
        local axisValue = axisMap[rotateAxisDropdown.Text] or "right"
        currentSelectedBlock:SetAttribute("axis", axisValue)
        -- Ensure numeric angular value
        local ang = tonumber(rotateAngularBox.Text) or 2.36
        currentSelectedBlock:SetAttribute("angular", ang)
        currentSelectedBlock:SetAttribute("loop", -1)
        currentSelectedBlock:SetAttribute("rotateType", "PersistentRotate")

        -- Enable simulation attributes expected by the loader/runtime
        currentSelectedBlock:SetAttribute("Simulate", true)
        currentSelectedBlock:SetAttribute("Timestep", 0)

        -- Create a ProxyToolTarget if one does not exist (used by load/save code)
        local proxyToolTarget = currentSelectedBlock:FindFirstChild("ProxyToolTarget")
        if not proxyToolTarget then
            proxyToolTarget = Instance.new("Part")
            proxyToolTarget.Name = "ProxyToolTarget"
            proxyToolTarget.Transparency = 1
            proxyToolTarget.Anchored = true
            proxyToolTarget.CanCollide = false
            proxyToolTarget.CanTouch = false
            proxyToolTarget.CastShadow = false
            proxyToolTarget.Parent = currentSelectedBlock
            local p = currentSelectedBlock.Position
            local o = currentSelectedBlock.Orientation
            proxyToolTarget.CFrame = CFrame.new(p) * CFrame.fromOrientation(math.rad(o.X), math.rad(o.Y), math.rad(o.Z))
            proxyToolTarget.Size = Vector3.new(2,2,2)
            proxyToolTarget.Color = Color3.new(0,0,0)
        end
        
        -- Clone and attach the Rotate behavior script
        local rotateScript = game.ServerStorage.TagFunctionScripts.Rotate:Clone()
        rotateScript.Parent = currentSelectedBlock
    end
    
    -- Resend obby JSON to backend and reload level to update block appearances
    sendParametersAndUpdateLevel()
    
    -- Reselect the block after level reload
    reselectBlockByName()
end

-- Function to toggle Translate behavior on/off
local function toggleTranslateBehavior()
    if not currentSelectedBlock then
        return
    end
    
    local hasBehavior = currentSelectedBlock:HasTag("PersistentTranslate")
    
    if hasBehavior then
        -- Remove the PersistentTranslate tag and any associated attributes/scripts
        print("Removing Translate behavior from block")
        currentSelectedBlock:RemoveTag("PersistentTranslate")
        currentSelectedBlock:SetAttribute("time", nil)
        currentSelectedBlock:SetAttribute("reverses", nil)
        currentSelectedBlock:SetAttribute("easingStyle", nil)
        currentSelectedBlock:SetAttribute("easingDirection", nil)
        currentSelectedBlock:SetAttribute("startPosition", nil)
        currentSelectedBlock:SetAttribute("endPosition", nil)
        currentSelectedBlock:SetAttribute("loop", nil)
        currentSelectedBlock:SetAttribute("translateType", nil)
        -- Remove the Translate script if it exists
        local translateScript = currentSelectedBlock:FindFirstChild("Translate")
        if translateScript then
            translateScript:Destroy()
        end
    else
        -- Add the PersistentTranslate tag and any associated attributes/scripts
        print("Adding Translate behavior to block")
        currentSelectedBlock:AddTag("PersistentTranslate")
        currentSelectedBlock:AddTag("Carry")
        currentSelectedBlock:SetAttribute("time", tonumber(translateTimeBox.Text))
        currentSelectedBlock:SetAttribute("reverses", true)
        currentSelectedBlock:SetAttribute("easingStyle", "Linear")
        currentSelectedBlock:SetAttribute("easingDirection", "Out")
        do
            local x, y, z = translateStartBox.Text:match("^%s*([-%d%.]+)%s*,%s*([-%d%.]+)%s*,%s*([-%d%.]+)%s*$")
            if x and y and z then
                local vec = Vector3.new(tonumber(x), tonumber(y), tonumber(z))
                currentSelectedBlock:SetAttribute("startPosition", HttpService:JSONEncode({vec.X, vec.Y, vec.Z}))
            else
                -- Fallback for invalid input
                currentSelectedBlock:SetAttribute("startPosition", HttpService:JSONEncode({0, 0, 0}))
            end
        end
        do
            local x, y, z = translateEndBox.Text:match("^%s*([-%d%.]+)%s*,%s*([-%d%.]+)%s*,%s*([-%d%.]+)%s*$")
            if x and y and z then
                local vec = Vector3.new(tonumber(x), tonumber(y), tonumber(z))
                currentSelectedBlock:SetAttribute("endPosition", HttpService:JSONEncode({vec.X, vec.Y, vec.Z}))
            else
                -- Fallback for invalid input
                currentSelectedBlock:SetAttribute("endPosition", HttpService:JSONEncode({0, 0, 0}))
            end
        end
        currentSelectedBlock:SetAttribute("loop", -1)
        currentSelectedBlock:SetAttribute("translateType", "PersistentTranslate")
        -- Add simulation controls expected by the loader/runtime
        currentSelectedBlock:SetAttribute("Simulate", true)
        currentSelectedBlock:SetAttribute("Timestep", 0)

        -- Ensure there is a ProxyToolTarget part (used by loader/save code)
        local proxyToolTarget = currentSelectedBlock:FindFirstChild("ProxyToolTarget")
        if not proxyToolTarget then
            proxyToolTarget = Instance.new("Part")
            proxyToolTarget.Name = "ProxyToolTarget"
            proxyToolTarget.Transparency = 1
            proxyToolTarget.Anchored = true
            proxyToolTarget.CanCollide = false
            proxyToolTarget.CanTouch = false
            proxyToolTarget.CastShadow = false
            proxyToolTarget.Parent = currentSelectedBlock
            local p = currentSelectedBlock.Position
            local o = currentSelectedBlock.Orientation
            proxyToolTarget.CFrame = CFrame.new(p) * CFrame.fromOrientation(math.rad(o.X), math.rad(o.Y), math.rad(o.Z))
            proxyToolTarget.Size = Vector3.new(2,2,2)
            proxyToolTarget.Color = Color3.new(0,0,0)
        end
        
        -- Clone and attach the Translate behavior script
        local translateScript = game.ServerStorage.TagFunctionScripts.Translate:Clone()
        translateScript.Parent = currentSelectedBlock

        -- Having "Carry" behavior is almost always what is desirable for translating blocks.
        local carryScript = game.ServerStorage.TagFunctionScripts.Carry:Clone()
        carryScript.Parent = currentSelectedBlock
    end
    
    -- Resend obby JSON to backend and reload level to update block appearances
    sendParametersAndUpdateLevel()
    
    -- Reselect the block after level reload
    reselectBlockByName()
end

-- Function to deactivate all behaviors in BlockEditor
local function deactivateAllBehaviors()
    lavaAddRemoveButton.Text = "Set Lava"
    lavaAddRemoveButton.BackgroundColor3 = Color3.fromRGB(50, 120, 220)
    
    -- Rotate frame
    rotateFrame.BackgroundColor3 = Color3.fromRGB(240, 240, 240)
    if rotateTitle then
        rotateTitle.Font = Enum.Font.SourceSans
    end
    rotateAddRemoveButton.Text = "Add"
    rotateAddRemoveButton.BackgroundColor3 = Color3.fromRGB(50, 120, 220)
    rotateAxisDropdown.Text = "X"
    rotateAngularBox.Text = "2.36"
    
    -- Translate frame
    translateFrame.BackgroundColor3 = Color3.fromRGB(240, 240, 240)
    if translateTitle then
        translateTitle.Font = Enum.Font.SourceSans
    end
    translateAddRemoveButton.Text = "Add"
    translateAddRemoveButton.BackgroundColor3 = Color3.fromRGB(50, 120, 220)
    translateStartBox.Text = "0,0,0"
    translateEndBox.Text = "10,0,0"
    translateTimeBox.Text = "1.0"
    
end

-- Connect add/remove buttons to toggle behaviors
lavaAddRemoveButton.MouseButton1Click:Connect(toggleLavaBehavior)
setSpawnButton.MouseButton1Click:Connect(setSpawnBehavior)
setGoalButton.MouseButton1Click:Connect(setGoalBehavior)
rotateAddRemoveButton.MouseButton1Click:Connect(toggleRotateBehavior)
translateAddRemoveButton.MouseButton1Click:Connect(toggleTranslateBehavior)

-- Connect selection changed to load behaviors
if selectionChangedConnection then
    selectionChangedConnection:Disconnect()
end

local bin_viz = {}
local goal_viz = {}
local intermediate_goal_viz = {}
local ckpt_viz = {}

selectionChangedConnection = SelectionService.SelectionChanged:Connect(function()
    local selected = SelectionService:Get()
    if #selected == 1 then
        if selected[1]:IsA("BasePart") then
            local selectedPart = selected[1]
            selectedBlockName = selectedPart.Name -- Store the block name
            -- Check if part is in current world
            if CollectionService:HasTag(selectedPart, tostring(currentWorldIdx)) then
                currentSelectedBlock = selectedPart
                loadBlockBehaviors(selectedPart)
            else
                -- Part not in current world, deactivate
                deactivateAllBehaviors()
            end
            if CollectionService:HasTag(selected[1], "GoExploreViz") then
                showGoExploreBins[1] = false
                local folder = Workspace:FindFirstChild("GoExploreBins")
                showOrHideGoExploreBins()
                selected[1].Parent = folder
                local ckpt_entry = ckpt_viz[selected[1].Position]   
                if ckpt_entry ~= nil then 
                    local trace = nil
                    if ckpt_entry[2] == nil then   
                        local url = backendUrl .. "/getTrajectoryAtSlot"
                        local slot_idx_table = {}
                        slot_idx_table.slot_idx = ckpt_entry[1]:GetAttribute("slot_idx")
                        local payload = HttpService:JSONEncode(slot_idx_table)
                        local success, response = pcall(HttpService.PostAsync, HttpService, url, payload)
                        if success then
                            local r = HttpService:JSONDecode(response)
                            trace = r.trace_viz[1]
                            ckpt_entry[2] = trace
                            print("REcieved trace", trace)
                        end
                    else
                        trace = ckpt_entry[2]
                    end
                    if trace ~= nil then
                        -- Clear existing trajectories
                        local obbyFolder = getFolderElseCreate(Workspace, "ObbyParts" .. tostring(currentWorldIdx))
                        for _, trajectoryType in {"Failure"} do
                            local currTrajectoryFolder = getFolderElseCreate(obbyFolder, "TrajectoryParts" .. trajectoryType)
                            clearFolderContents(currTrajectoryFolder)
                        end
                    end
                    local result = {
                        trace_viz = {},
                        success_trace_viz = {},
                        failure_trace_viz = {trace},
                        human_trace_viz = {}
                    }
                    
                    -- Get the name of the selected checkpoint part and convert it into frame number
                    local checkpointName = selected[1].Name
                    local frameNumber = tonumber(checkpointName:match("%d+"))
                    -- Set the frame number in the result for trajectory visualization
                    simulationTimestep = frameNumber
                    drawTrajectories(result)
                    inSuccessTrajectoryMode[1] = false
                    inFailureTrajectoryMode[1] = true
                    showSuccessOrFailureTrajectories()
                end
            else
                -- If selection moved off a GoExploreViz checkpoint, restore bin visualization
                if showGoExploreBins[1] == false and buttonShowGoExploreBins[1] == true then
                    showGoExploreBins[1] = true
                    showOrHideGoExploreBins()
                end
            end
        else
            -- No valid selection, deactivate all behaviors
            currentSelectedBlock = nil
            selectedBlockName = nil
            deactivateAllBehaviors()
            -- If selection cleared or non-part selected, restore go-explore bins
            if showGoExploreBins[1] == false and buttonShowGoExploreBins[1] == true then
                showGoExploreBins[1] = true
                showOrHideGoExploreBins()
            end
        end
    elseif showGoExploreBins[1] == false and buttonShowGoExploreBins[1] == true then
        -- Empty selection should also restore the bins.
        showGoExploreBins[1] = true
        showOrHideGoExploreBins()
    end
end)

function addCheckboxToFrame(position, label, activated, connectedFunction, frame)
    local checkboxTable = {}

    local textLabel = Instance.new("TextLabel")
    textLabel.Text = label
    textLabel.Size = UDim2.new(0, analysisFrame.AbsoluteSize.X, 0, analysisFrame.AbsoluteSize.Y * 0.5)
    textLabel.Position = position
    textLabel.BackgroundTransparency = 1
    textLabel.TextSize = 16
    textLabel.Font = Enum.Font.SourceSansBold
    textLabel.TextXAlignment = Enum.TextXAlignment.Left
    textLabel.Visible = true
    textLabel.Parent = frame

    local checkboxFrame = Instance.new("Frame")
    checkboxFrame.Size = UDim2.new(0, 12, 0, 12)
    checkboxFrame.Position = position + UDim2.new(0, -60, 0, 25)
    checkboxFrame.BackgroundColor3 = Color3.fromRGB(255, 255, 255)
    checkboxFrame.BorderSizePixel = 1
    checkboxFrame.BorderColor3 = Color3.fromRGB(27, 42, 53)
    checkboxFrame.Visible = true -- Make it initially invisible
    checkboxFrame.Parent = frame

    -- Create the button to make it clickable
    local button = Instance.new("TextButton")
    button.Size = UDim2.new(1, 0, 1, 0)
    button.BackgroundTransparency = 1
    button.Text = ""
    button.Parent = checkboxFrame

    -- Create the checkmark (initially invisible if defaultValue is false)
    local checkmark = Instance.new("TextLabel")
    checkmark.Size = UDim2.new(1, 0, 1, 0)
    checkmark.BackgroundTransparency = 1
    checkmark.Text = "✓"
    checkmark.TextSize = 18
    checkmark.TextColor3 = Color3.fromRGB(0, 0, 0)
    checkmark.Font = Enum.Font.SourceSansBold
    checkmark.Visible = activated[1]
    checkmark.Parent = checkboxFrame

    -- Connect the checkbox click events
    button.MouseButton1Click:Connect(function()
        activated[1] = not activated[1]
        checkmark.Visible = activated[1]
        connectedFunction()
    end)

    checkboxTable["label"] = textLabel
    checkboxTable["checkmark"] = checkmark
    checkboxTable["activated"] = activated

    return checkboxTable
end

function updatePlayerAndDynamicObject()
    if showPlayerAndDynamicObjects[1] then
        local player = PlayerCloneModule.getCurrentPlayer()
        print("Showing player:", player.Character)
        if player then
            -- Show player
            PlayerCloneModule.showClone(player.Character)
        end

        -- Find all objects with "Unanchored" tag and show them
        local dynamicObjects = CollectionService:GetTagged("Unanchored")
        for _, obj in pairs(dynamicObjects) do
            if obj:IsA("BasePart") then
                obj.Transparency = 0
                obj.CanCollide = true
                obj.Anchored = false
            end
        end
    else
        local player = PlayerCloneModule.getCurrentPlayer()
        print("Hiding player:", player.Character)
        if player then
            -- Hide player
            PlayerCloneModule.hideClone(player.Character)
        end

        -- Find all objects with "Unanchored" tag and hide them
        local dynamicObjects = CollectionService:GetTagged("Unanchored")
        for _, obj in pairs(dynamicObjects) do
            if obj:IsA("BasePart") then
                obj.Transparency = 1
                obj.CanCollide = false
                obj.Anchored = true
            end
        end
    end
end

-- Function to replay a trajectory
function replayTrajectories(trajectoryFolders)
    -- Reset world to initial state
    resetObstacleInstance(currentWorldIdx)
    local traj_start_step = 0
    -- Make all parts invisible.
    local clones = {}
    local dynamic_block_clones = {}
    for _, trajectoryFolder in pairs(trajectoryFolders) do
        for i, part in pairs(trajectoryFolder:GetChildren()) do
            toggleTrajectoryPart(part, not inLivePlaybackMode[1])
            local step = part:GetAttribute("timestep")
            if step ~= nil then
                if traj_start_step == 0 then
                    traj_start_step = step
                end
                traj_start_step = math.min(traj_start_step, step)
            end
        end
        local clone = trajectoryFolder:FindFirstChild(currAgentName)
        if inLivePlaybackMode[1] and clone then
            if not clone then
                print("Clone is nil")
            end
            PlayerCloneModule.showClone(clone)
        end
        local dynamic_block_clone = trajectoryFolder:FindFirstChild("BlockClone")
        if inLivePlaybackMode[1] and dynamic_block_clone then
            dynamic_block_clone.Transparency = 0.5
        end
        table.insert(clones, clone)
        table.insert(dynamic_block_clones, dynamic_block_clone)
    end

    local function getTypes(trajectoryFolder, type, tag)
        local parts = {}
        for _, part in pairs(trajectoryFolder:GetChildren()) do
            if part:IsA(type) and CollectionService:HasTag(part, tag) then
                table.insert(parts, part)
            end
        end
        return parts
    end

    local stepJob = nil
    if inLivePlaybackMode[1] then
        -- Disable player and dynamic_object display
        showPlayerAndDynamicObjects[1] = false
        updatePlayerAndDynamicObject()

        simulationTimestep = traj_start_step
        setSimulationTimestep(simulationTimestep)
        stepJob = RunService.Heartbeat:Connect(function()
            local partsToShow = {}
            local blockPartsToShow = {}
            local avg_dt = 0.0
            for traj_idx, trajectoryFolder in pairs(trajectoryFolders) do
                -- Must use "getTypes" here because it returns the trajectory points in order.
                local agent_pos = getTypes(trajectoryFolder, "Part", "AgentPos")
                local block_pos = getTypes(trajectoryFolder, "Part", "BlockPos")

                local clone = clones[traj_idx]
                local block_clone = dynamic_block_clones[traj_idx]
                if #agent_pos >= (simulationTimestep + 1 - traj_start_step) then
                    -- Set parts visible based on the simulation timestep
                    for idx, part in pairs(agent_pos) do
                        toggleTrajectoryPart(part, idx <= (simulationTimestep + 1 - traj_start_step))
                    end
                    for idx, part in pairs(block_pos) do
                        toggleTrajectoryPart(part, idx <= (simulationTimestep + 1 - traj_start_step))
                    end
                    local agent_part = agent_pos[(simulationTimestep + 1 - traj_start_step)]
                    local block_part = block_pos[(simulationTimestep + 1 - traj_start_step)]
                    table.insert(partsToShow, agent_part)
                    table.insert(blockPartsToShow, block_part)
                    PlayerCloneModule.moveClone(clone, agent_part.Position + Vector3.new(0, submissionTrajectoryOffset, 0), false)
                    if block_clone ~= nil then
                        block_clone.Position = block_part.Position
                    end

                elseif #agent_pos > 0 and #agent_pos == (simulationTimestep - traj_start_step) and agent_pos[1].Color == ColorSequence.new(Color3.new(1, 0, 0)) then
                    PlayerCloneModule.killClone(clone)
                end
            end
            if #partsToShow == 0 then
                -- Reset world to initial state
                resetObstacleInstance(currentWorldIdx)
                for _, trajectoryFolder in pairs(trajectoryFolders) do
                    for _, part in pairs(getTypes(trajectoryFolder, "Part", "AgentPos")) do
                        toggleTrajectoryPart(part, false)
                    end
                end
                simulationTimestep = traj_start_step
                setSimulationTimestep(simulationTimestep)
            elseif not simulationPaused[1] then
                simulationTimestep += 1
            end 
    	end)
    else
        -- Enable player and dynamic_object display
        showPlayerAndDynamicObjects[1] = true
        updatePlayerAndDynamicObject()
        -- Pause sim and set the timestep.
        simulationPaused[1] = true
        playOrPauseSimulation()
        wait(0.1) -- Wait to allow the sim to correctly set timesteps on objects.
        setSimulationTimestep(simulationTimestep)
        wait(0.1) -- Wait to allow the sim to correctly set timesteps on objects.
    end
    return stepJob
end

function showSuccessOrFailureTrajectories()
    local trajectoryFolders = {}

    local totalSuccessTrajectories = 0
    local totalFailureTrajectories = 0

    -- Hide all existing trajectories and count the number of success and failure trajectories.
    local obbyFolder = getFolderElseCreate(Workspace, "ObbyParts" .. tostring(currentWorldIdx))
    for _, trajectoryType in {"Success", "Human", "Failure"} do
        local currTrajectoryFolder = getFolderElseCreate(obbyFolder, "TrajectoryParts" .. trajectoryType)
        hideTrajectories(currTrajectoryFolder)
        if trajectoryType == "Success" then
            totalSuccessTrajectories = #currTrajectoryFolder:GetChildren()
        end
        if trajectoryType == "Failure" then
            totalFailureTrajectories = #currTrajectoryFolder:GetChildren()
        end
    end

    print("Total successes: ", totalSuccessTrajectories)
    print("Total failures: ", totalFailureTrajectories)

    -- global numShownTrajectories.
    print("Total shown trajectories:", numShownTrajectories)
    local maxSuccessTrajectories = math.min(numShownTrajectories, totalSuccessTrajectories)
    local maxFailureTrajectories = math.min(numShownTrajectories, totalFailureTrajectories)

    print("Max success trajectories: ", maxSuccessTrajectories)
    print("Max failure trajectories: ", maxFailureTrajectories)

    if inSuccessTrajectoryMode[1] then
        local obbyFolder = getFolderElseCreate(Workspace, "ObbyParts" .. tostring(currentWorldIdx))
        local currTrajectoryFolder = getFolderElseCreate(obbyFolder, "TrajectoryPartsSuccess")

        showTrajectories(currTrajectoryFolder, maxSuccessTrajectories, activeTrajectoryIdx)
        for i, trajectoryInstanceFolder in currTrajectoryFolder:GetChildren() do
            if i <= maxSuccessTrajectories then
                if activeTrajectoryIdx == -1 or activeTrajectoryIdx == i then
                    table.insert(trajectoryFolders, trajectoryInstanceFolder)
                end
            end
        end

        local obbyFolder = getFolderElseCreate(Workspace, "ObbyParts" .. tostring(currentWorldIdx))
        local currTrajectoryFolder = getFolderElseCreate(obbyFolder, "TrajectoryPartsHuman")

        showTrajectories(currTrajectoryFolder)
        for _, trajectoryInstanceFolder in currTrajectoryFolder:GetChildren() do
            table.insert(trajectoryFolders, trajectoryInstanceFolder)
        end
    end

    if inFailureTrajectoryMode[1] then
        local obbyFolder = getFolderElseCreate(Workspace, "ObbyParts" .. tostring(currentWorldIdx))
        local currTrajectoryFolder = getFolderElseCreate(obbyFolder, "TrajectoryPartsFailure")

        showTrajectories(currTrajectoryFolder, maxFailureTrajectories, activeTrajectoryIdx)
        for i, trajectoryInstanceFolder in currTrajectoryFolder:GetChildren() do
            if i <= maxFailureTrajectories then
                if activeTrajectoryIdx == -1 or activeTrajectoryIdx == i then
                    table.insert(trajectoryFolders, trajectoryInstanceFolder)
                end
            end
        end
    end

    if trajectoryStepJob then
        trajectoryStepJob:Disconnect()
        trajectoryStepJob = nil
    end

    if inSuccessTrajectoryMode[1] or inFailureTrajectoryMode[1] then
        trajectoryStepJob = replayTrajectories(trajectoryFolders)
    end

end

local function setCameraPosition(rotationDegrees)
    local spawnPos = Vector3.new(0,0,0)
    local goalPos = Vector3.new(0,0,0)
    for i, part in CollectionService:GetTagged(currentWorldIdx) do
        if CollectionService:HasTag(part, "Spawn") then
            spawnPos = part.Position
        end
        if CollectionService:HasTag(part, "Checkpoint") and CollectionService:HasTag(part, currentWorldIdx + 1) then
            goalPos = part.Position
        end
    end

    local midpoint = 0.5 * (spawnPos + goalPos)

    local offset = math.max(50, (goalPos - spawnPos).Magnitude * 0.5)
    local cameraOffsetFrame = nil
    cameraOffsetFrame = CFrame.lookAlong(midpoint, goalPos - spawnPos, Vector3.new(0,1,0)) + (CFrame.Angles(0,0,math.rad(rotationDegrees)) * CFrame.new(0, offset, 0)).Position
    workspace.CurrentCamera.CFrame = cameraOffsetFrame
    workspace.CurrentCamera.Focus = CFrame.new(midpoint)
end
viewCameraButton = createGUIButton("View", UDim2.new(0, 0, 0, 30), UDim2.new(0, 0, 0, 130), contentFrame, function() setCameraPosition(45) end)
topCameraButton = createGUIButton("Top", UDim2.new(0, 0, 0, 30), UDim2.new(0, 0, 0, 130), contentFrame, function() setCameraPosition(0.01) end)
sideCameraButton = createGUIButton("Side", UDim2.new(0, 0, 0, 30), UDim2.new(0, 0, 0, 130), contentFrame, function() setCameraPosition(90) end)

--- Folder helper functions
function getFolderElseCreate(parent, name)
    local folder = parent:FindFirstChild(name)
    if folder == nil then
        folder = Instance.new("Folder", parent)
        folder.Name = name
    end
    return folder
end

function clearFolderContents(folder)
    for _, child in pairs(folder:GetChildren()) do
        if not child:IsA("ProceduralModel") then
            child:Destroy()
        end
    end
end

function _deepcopy_table(t)
    local copy = {}
    for k in t do
        if type(t[k]) == "table" then
            copy[k] = _deepcopy_table(t[k])
        else
            copy[k] = t[k]
        end
    end
    return copy
end

-- Update the level geometry in a given folder from a json description and an option list of json diffs.
function updateObstacleInstance(obbyFolder, level_json, diffs_json)
    -- Destroy previous parts
    clearFolderContents(obbyFolder)
    local goExploreBins = workspace:FindFirstChild("GoExploreBins")
    if goExploreBins then
        clearFolderContents(goExploreBins)
    end

    local combined_json = _deepcopy_table(level_json)
    print("level_json", level_json)

    -- Apply the diffs
    for block_name, diff in pairs(diffs_json) do
        if level_json[block_name] then
            if diff["deleted"] then
                combined_json[block_name] = nil
            else
                for i = 1, 3, 1 do
                    combined_json[block_name]["position"][i] = level_json[block_name]["position"][i] + diff["position"][i] 
                    combined_json[block_name]["orientation"][i] = level_json[block_name]["orientation"][i] + diff["orientation"][i] 
                    combined_json[block_name]["size"][i] = level_json[block_name]["size"][i] * diff["size"][i]
                end
                combined_json[block_name]["behaviors"] = diff["behaviors"]
                if diff["type"] then
                    combined_json[block_name]["type"] = diff["type"]
                end
            end
        else
            -- Check for created blocks. If the DSL parameter would get rid of a block
            -- (e.g. n_blocks was decreased) it happens.
            if diff["created"] then
                combined_json[block_name] = diff
            end
        end
    end

    -- Load in new parts
    local obbyParts = LoadObbyFunctions.createObbyFromJson(combined_json, nil, 1)
    for i, obbyPart in pairs(obbyParts) do
        obbyPart.Parent = obbyFolder
    end

    return
end

-- Update the level geometry in the workspace from a JSON
function resetObstacleInstance(sectionIdx)

    local obbyFolder = getFolderElseCreate(Workspace, "ObbyParts" .. tostring(currentWorldIdx))

    simulationPaused[1] = true
    playOrPauseSimulation()
    setSimulationTimestep(0) -- Always save levels in the starting orientation.
    wait(0.1) -- Wait to allow the sim to correctly set timesteps on objects.

    local currentWorkspaceJson = LoadObbyFunctions.createJsonFromObbySection(sectionIdx)
    print(currentWorkspaceJson)

    -- Destroy previous parts
    for _, child in pairs(obbyFolder:GetChildren()) do
        if child:IsA("BasePart") then
            child:Destroy()
        end
    end
    -- Need to destroy this after so behavior scripts don't lock up.
    obbyFolder:FindFirstChild("States"):Destroy()
    -- Load in new parts
    local obbyParts = LoadObbyFunctions.createObbyFromJson(currentWorkspaceJson, nil, sectionIdx)
    for i, obbyPart in pairs(obbyParts) do
        obbyPart.Parent = obbyFolder
    end

    -- Get all dynamic objects with "Unanchored" tag.
    local dynamicObjects = CollectionService:GetTagged("Unanchored")
    if not showPlayerAndDynamicObjects[1] then
        -- If the option to show dynamic objects is disabled, set them to invisible.
        for _, obj in pairs(dynamicObjects) do
            if obj:IsA("BasePart") then
                obj.Transparency = 1
                obj.CanCollide = false
                obj.Anchored = true
            end
        end
    end

    simulationPaused[1] = false
    playOrPauseSimulation()

    return
end

-- Request all the level types from the backend
function fetchLevelOptions()
    local success, response = pcall(HttpService.GetAsync, HttpService, backendUrl .. "/requestAllLevels")

    if success then
        local result = HttpService:JSONDecode(response)
        return result.levels
    else
        print(response)
        print("Error loading options")
    end
end

-- Request all the level types from the backend
function fetchAgentOptions()
    local success, response = pcall(HttpService.GetAsync, HttpService, backendUrl .. "/requestAllAgents")

    if success then
        local result = HttpService:JSONDecode(response)
        return result.agents
    else
        print("Error loading options")
    end
end

-- Request all the level types from the backend
function fetchLoadLevelOptions()
    local success, response = pcall(HttpService.GetAsync, HttpService, backendUrl .. "/requestSavedLevels")

    if success then
        local result = HttpService:JSONDecode(response)
        return result.levels
    else
        print("Error loading options")
    end
end

function fetchLoadQueryOptions()
    local success, response = pcall(HttpService.GetAsync, HttpService, backendUrl .. "/requestSavedQueries")

    if success then
        local result = HttpService:JSONDecode(response)
        return result.levels
    else
        print("Error loading options")
    end
end

-- Request the parameters for a given level from the backend
--  (Note this updates the state of the UI)
function requestLevelParameters()
    local url = backendUrl .. "/requestLevelType/" .. HttpService:UrlEncode(currentLevelType[currentWorldIdx])
    local success, response = pcall(HttpService.GetAsync, HttpService, url)

    if success then
        local result = HttpService:JSONDecode(response)
        return result.parameters, result.level_json
    else
        return {}
    end
end

function requestSavedLevelParameters(level_path)
    local url = backendUrl .. "/requestSavedLevel/" .. HttpService:UrlEncode(level_path)
    local success, response = pcall(HttpService.GetAsync, HttpService, url)

    if success then
        local result = HttpService:JSONDecode(response)
        return result.level_type, result.parameters, result.level_json, result.dsl_json
    else
        return {}
    end
end

function requestSavedQueryParameters(query_path)
    local url = backendUrl .. "/requestSavedQuery/" .. HttpService:UrlEncode(query_path)
    local success, response = pcall(HttpService.GetAsync, HttpService, url)

    if success then
        local result = HttpService:JSONDecode(response)
        return result.level_type, 
        result.parameters, 
        result.level_json, 
        result.dsl_json, 
        result.max_traj_steps, 
        result.max_traj_jumps, 
        result.traj_start_step,
        result.bounds_position,
        result.bounds_size
    else
        return {}
    end
end

function computeDiff(original, current)
    -- Used to detect no change in an obstacle parameter.
    local EPS = 0.001
    local function snapToValue(x, value)
        if math.abs(x - value) < EPS then
            return value
        end
        return x
    end
    local diffs = {}
    for block_name in original do
        local diff = {}
        diff["created"] = false
        if current[block_name] then
            diff["position"] = {}
            diff["orientation"] = {}
            diff["size"] = {}
            for i = 1, 3, 1 do
                diff["position"][i] = snapToValue(current[block_name]["position"][i] - original[block_name]["position"][i], 0)
                diff["orientation"][i] = snapToValue(current[block_name]["orientation"][i] - original[block_name]["orientation"][i], 0)
                diff["size"][i] = snapToValue(current[block_name]["size"][i] / original[block_name]["size"][i], 1)
            end
            diff["deleted"] = false
            diff["behaviors"] = current[block_name]["behaviors"]
            diff["type"] = current[block_name]["type"]
        else
            -- The original block name was not found in current, meaning it was deleted.
            diff["deleted"] = true
        end
        diffs[block_name] = diff
    end

    for block_name in current do
        if not original[block_name] then
            -- A block was created.
            diffs[block_name] = current[block_name]
            diffs[block_name]["created"] = true
        end
    end
    return diffs
end

-- Prepare the level + params payload for sending to the backend
-- Includes:
-- DSL Parameters
-- Level type (ThickPathJumps, LavaSpinner, etc.)
-- JSON of the current level as it exists in the Workspace with local edits.
-- The diffs between the current level and the vanilla DSL level, computed over the JSONs.
function prepareLevelParamsPayload()
    local currentWorkspaceJson = LoadObbyFunctions.createJsonFromObbySection(currentWorldIdx)

    local payload = {
        parameters = currentParameters[currentWorldIdx],
        level_type = currentLevelType[currentWorldIdx],
        level_json = currentWorkspaceJson,
        diffs = computeDiff(currentLevelJson[currentWorldIdx], currentWorkspaceJson)
    }
    return payload
end

-- Send the current slider parameters to backend (then update the level geometry)
function sendParametersAndUpdateLevel()
    local payload = prepareLevelParamsPayload()

    diffs = payload["diffs"]
    -- Send the current parameters to backend
    local jsonData = HttpService:JSONEncode(payload)
	local success, response = pcall(HttpService.PostAsync, HttpService, backendUrl .. "/sendCurrentParameters", jsonData)

    if not success then
        print("Failed on /sendCurrentParameters in sendParametersAndUpdateLevel")
    end

    -- Get the new level json and update
    local result = HttpService:JSONDecode(response)
    local levelJson = result.level_json

    local obbyFolder = getFolderElseCreate(Workspace, "ObbyParts" .. tostring(currentWorldIdx))
    updateObstacleInstance(obbyFolder, levelJson, diffs)

    return levelJson
end

function toggleTrajectoryPart(part, enabled)
    if part:IsA("Part") then
        if enabled then
            part.Transparency = 0.2
        else
            part.Transparency = 1
        end
    end 
end

function showTrajectories(trajectoryFolder, maxTrajectories, trajectoryIdx)
    for i, singleTrajectoryFolder in pairs(trajectoryFolder:GetChildren()) do
        if i <= maxTrajectories then
            if singleTrajectoryFolder:IsA("Folder") and singleTrajectoryFolder.Name:find("Trajectory") then
                if trajectoryIdx == -1 or singleTrajectoryFolder.Name:find(tostring(trajectoryIdx)) then
                    for _, part in pairs(singleTrajectoryFolder:GetChildren()) do
                        if part:IsA("Part") then
                            toggleTrajectoryPart(part, true)
                        end
                    end
                end
            end
        end
    end
end

function hideTrajectories(trajectoryFolder)
    for _, singleTrajectoryFolder in pairs(trajectoryFolder:GetChildren()) do
        if singleTrajectoryFolder:IsA("Folder") and singleTrajectoryFolder.Name:find("Trajectory") then
            for _, part in pairs(singleTrajectoryFolder:GetChildren()) do
                toggleTrajectoryPart(part, false)
                if part.Name == currAgentName then
                    PlayerCloneModule.hideClone(part)
                end
            end
        end
    end
end

function drawTrajectories(result, starting_timestep)
    -- Draw the trajectories
    local obbyFolder = getFolderElseCreate(Workspace, "ObbyParts" .. tostring(currentWorldIdx))
    local trajectorySuccessFolder = getFolderElseCreate(obbyFolder, "TrajectoryPartsSuccess")
    local trajectoryFailureFolder = getFolderElseCreate(obbyFolder, "TrajectoryPartsFailure")
    local trajectoryHumanFolder = getFolderElseCreate(obbyFolder, "TrajectoryPartsHuman")
    local trajectories = result.trace_viz
    local successTrajectories = result.success_trace_viz
    local failureTrajectories = result.failure_trace_viz
    local humanTrajectories = result.human_trace_viz
    local function makeTrajectory(trajectory, t_idx, traj_color, traj_folder)
        local trajectorySampleFolder = getFolderElseCreate(traj_folder, "TrajectoryParts"..t_idx)
        local x = trajectory.x
        local y = trajectory.y
        local z = trajectory.z
        local block_x = trajectory.block_x
        local block_y = trajectory.block_y
        local block_z = trajectory.block_z
        local t = trajectory.t
        local action = trajectory.action
        local reward = trajectory.reward
        local value = trajectory.value
        local action_dist = trajectory.action_dist

        -- Make a clone of the player
        local clone = PlayerCloneModule.createClone({
            highlight = false,            -- Add highlight effect
            highlightColor = traj_color,  -- Cyan highlight
            outlineColor = traj_color,
            name = tostring(currAgentName),
            showNameLabel = false,
            fillTransparency = 0,
            outlineTransparency = 0,
            parent = trajectorySampleFolder,
            anchored = true -- Setting this value to false will trigger touched events.
        })
        PlayerCloneModule.hideClone(clone)

        local block_copy = nil
        -- Create the block part if necessary
        if block_x[1] ~= 0 or block_y[1] ~= 0 or block_z[1] ~= 0 then
            local dynamic_block = CollectionService:GetTagged("Unanchored")[1]
            if dynamic_block == nil then
                print("Trajectory has dynamic object data but dynamic object data not found.")
            else
                block_copy = Instance.new("Part")
                block_copy.Color = dynamic_block.Color
                block_copy.Transparency = 0.5
                block_copy.Shape = dynamic_block.Shape
                block_copy.Size = dynamic_block.Size
                block_copy.Parent = trajectorySampleFolder
                block_copy.Name = "BlockClone"
                block_copy.Anchored = true
                block_copy.CanCollide = false
                block_copy.Position = dynamic_block.Position
            end
        end

        if inSuccessTrajectoryMode[1] or inFailureTrajectoryMode[1] then
            if inLivePlaybackMode[1] then 
                PlayerCloneModule.showClone(clone)
            end
            if block_copy ~= nil then
                block_copy.Transparency = 0.5
            end
        else
            PlayerCloneModule.hideClone(clone)
            if block_copy ~= nil then
                block_copy.Transparency = 1
            end
        end
        local prev_part = nil
        local avg_value = 0
        local avg_advantage = 0
        local avg_action_probs = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}

        -- Changing from 1 will break live playback visualization because the obstacle animation speed will not increase.
        for i = 1, #x, 1 do -- Taking a larger step will subsample trajectories so they all appear onscreen when being drawn with beams, or in general for performance.
            local worldOffset = worldTranslationOffsets[currentWorldIdx]
            -- Create two parts at the end points
            local part1 = Instance.new("Part")
            part1.Locked = true
            part1.Position = Vector3.new(x[i], y[i] - submissionTrajectoryOffset, z[i]) + worldOffset
            part1.Size = Vector3.new(0.2, 0.2, 0.2)
            part1.Color = traj_color
            if i == #x then
                part1.Size = Vector3.new(0.5, 0.5, 0.5)
                part1.Color = Color3.new(traj_color.R * 2, traj_color.G * 2, traj_color.B * 2)
            end

            part1.CastShadow = false
            part1.CanTouch = false
            part1.CanQuery = false
            part1.Anchored = true
            part1.Material = Enum.Material.Neon
            part1.CanCollide = false
            if inSuccessTrajectoryMode[1] or inFailureTrajectoryMode[1] then
                part1.Transparency = 0.2
            else
                part1.Transparency = 1
            end

            -- Set trajectory metadata
            if starting_timestep == nil then
                starting_timestep = 0
            end
            part1:SetAttribute("timestep", t[i] + starting_timestep)
            part1:SetAttribute("value", value[i])

            local advantage = 0
            if i > 1 then
                advantage = reward[i] + 0.998 * value[i] - value[i-1]
            end
            part1:SetAttribute("advantage", advantage)
            avg_advantage += advantage
            avg_value += value[i]
            for j = 1, #action_dist[i], 1 do
                avg_action_probs[j] += action_dist[i][j]
            end

            part1:SetAttribute("reward", string.format("%.6f", reward[i]))
            part1:SetAttribute("action", "Move="..string.format("%d", action[i][1])..", Dir="..string.format("%d", action[i][2])..", Jump="..string.format("%d", action[i][4]))
            part1:SetAttribute("Move", string.format("%.3f", action_dist[i][1])..", "..string.format("%.3f", action_dist[i][2]))

            local dirString = ""
            for j = 1, 8, 1 do
                dirString = dirString..string.format("%.3f", action_dist[i][2 + j])..", "
            end
            part1:SetAttribute("Direction", dirString)
            part1:SetAttribute("Rotate", 
            string.format("%.3f", action_dist[i][11])..", "..
            string.format("%.3f", action_dist[i][12])..", "..
            string.format("%.3f", action_dist[i][13]))
            part1:SetAttribute("Jump", string.format("%.6f", action_dist[i][14])..","..string.format("%.6f", action_dist[i][15]))

            part1.Parent = trajectorySampleFolder
            CollectionService:AddTag(part1, "AgentPos")
            -- Attach the TrajectoryPart script
            local trajectoryScript = game.ServerStorage.TagFunctionScripts.TrajectoryPart:Clone()
            trajectoryScript.Parent = part1

            if block_copy ~= nil then
                local part2 = Instance.new("Part")
                part2.Locked = true
                part2.Position = Vector3.new(block_x[i], block_y[i], block_z[i]) + worldOffset
                part2.Size = Vector3.new(0.2, 0.2, 0.2)
                part2.Color = block_copy.Color
                if i == #x then
                    part2.Size = Vector3.new(0.5, 0.5, 0.5)
                    part2.Color = block_copy.Color
                end

                part2.CastShadow = false
                part2.CanTouch = false
                part2.CanQuery = false
                part2.Anchored = true
                part2.Material = Enum.Material.Neon
                part2.CanCollide = false
                if inSuccessTrajectoryMode[1] or inFailureTrajectoryMode[1] then
                    part2.Transparency = 0.2
                else
                    part2.Transparency = 1
                end
                part2.Parent = trajectorySampleFolder
                CollectionService:AddTag(part2, "BlockPos")
                -- Attach the TrajectoryPart script
                local trajectoryScript = game.ServerStorage.TagFunctionScripts.TrajectoryPart:Clone()
                trajectoryScript.Parent = part2
            end
        end
        local AvgValue = Instance.new("NumberValue")
        AvgValue.Name = "Average Value"
        AvgValue.Value = avg_value / #x
        AvgValue.Parent = trajectorySampleFolder

        local AvgAdvantage = Instance.new("NumberValue")
        AvgAdvantage.Name = "Average Advantage"
        AvgAdvantage.Value = avg_advantage / #x
        AvgAdvantage.Parent = trajectorySampleFolder

        local AvgHighestActionProb = Instance.new("StringValue")
        AvgHighestActionProb.Name = "Average Action Probs"
        for j = 1, #avg_action_probs, 1 do
            avg_action_probs[j] /= #x
        end
        AvgHighestActionProb.Value = HttpService:JSONEncode(avg_action_probs)
        AvgHighestActionProb.Parent = trajectorySampleFolder
        return trajectorySampleFolder
    end
    
    -- All trajectory colors are slightly darker to better contrast against saturated obstacle colors.
    -- Each has slight variance to make individual ones stand out. 
    local t_idx = #trajectorySuccessFolder:getChildren() + 1
    local global_avg_value = 0
    local global_avg_advantage = 0
    local global_avg_action_probs = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}
    for color_i, trajectory in pairs(successTrajectories) do
        local sampleFolder = makeTrajectory(trajectory, t_idx, Color3.new(0,.5,0), trajectorySuccessFolder)
        global_avg_value += sampleFolder:FindFirstChild("Average Value").Value
        global_avg_advantage += sampleFolder:FindFirstChild("Average Advantage").Value
        local action_probs = HttpService:JSONDecode(sampleFolder:FindFirstChild("Average Action Probs").Value)
        for j = 1, #action_probs, 1 do
            global_avg_action_probs[j] += action_probs[j]
        end
        t_idx += 1
    end
    local GlobalAvgValue = Instance.new("NumberValue")
    GlobalAvgValue.Name = "Average Value"
    GlobalAvgValue.Value = global_avg_value / math.max(#successTrajectories, 1)
    GlobalAvgValue.Parent = trajectorySuccessFolder
    local GlobalAvgAdvantage = Instance.new("NumberValue")
    GlobalAvgAdvantage.Name = "Average Advantage"
    GlobalAvgAdvantage.Value = global_avg_advantage / math.max(#successTrajectories, 1)
    GlobalAvgAdvantage.Parent = trajectorySuccessFolder
    local GlobalAvgActionProbs = Instance.new("StringValue")
    GlobalAvgActionProbs.Name = "Average Action Probs"
    for j = 1, #global_avg_action_probs, 1 do
        global_avg_action_probs[j] /= math.max(#successTrajectories, 1)
    end
    GlobalAvgActionProbs.Value = HttpService:JSONEncode(global_avg_action_probs)
    GlobalAvgActionProbs.Parent = trajectorySuccessFolder

    t_idx = #trajectoryFailureFolder:getChildren() + 1
    global_avg_value = 0
    global_avg_advantage = 0
    global_avg_action_probs = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}
    for _, trajectory in pairs(failureTrajectories) do
        local sampleFolder = makeTrajectory(trajectory, t_idx, Color3.new(.5, .5, 0), trajectoryFailureFolder)
        global_avg_value += sampleFolder:FindFirstChild("Average Value").Value
        global_avg_advantage += sampleFolder:FindFirstChild("Average Advantage").Value
        local action_probs = HttpService:JSONDecode(sampleFolder:FindFirstChild("Average Action Probs").Value)
        for j = 1, #action_probs, 1 do
            global_avg_action_probs[j] += action_probs[j]
        end
        t_idx += 1
    end
    local GlobalAvgValue = Instance.new("NumberValue")
    GlobalAvgValue.Name = "Average Value"
    GlobalAvgValue.Value = global_avg_value / math.max(#failureTrajectories, 1)
    GlobalAvgValue.Parent = trajectoryFailureFolder
    local GlobalAvgAdvantage = Instance.new("NumberValue")
    GlobalAvgAdvantage.Name = "Average Advantage"
    GlobalAvgAdvantage.Value = global_avg_advantage / math.max(#failureTrajectories, 1)
    GlobalAvgAdvantage.Parent = trajectoryFailureFolder
    local GlobalAvgActionProbs = Instance.new("StringValue")
    GlobalAvgActionProbs.Name = "Average Action Probs"
    for j = 1, #global_avg_action_probs, 1 do
        global_avg_action_probs[j] /= math.max(#failureTrajectories, 1)
    end
    GlobalAvgActionProbs.Value = HttpService:JSONEncode(global_avg_action_probs)
    GlobalAvgActionProbs.Parent = trajectoryFailureFolder

    t_idx = #trajectoryHumanFolder:getChildren() + 1
    for _, trajectory in pairs(humanTrajectories) do
        makeTrajectory(trajectory, t_idx, Color3.new(0.9,0,0.9), trajectoryHumanFolder)
        t_idx += 1
    end
end

local numTrajectoriesSlider = nil
function requestSolutionTrajectory()
    -- Pause sim and set the timestep.
    simulationPaused[1] = true
    playOrPauseSimulation()
    setSimulationTimestep(searchStartTimestep)
    wait(0.1) -- Wait to allow the sim to correctly set timesteps on objects.

    PhysicsService:RegisterCollisionGroup("GoExploreViz")
    -- Enable failure trajectory mode when "Find Solution" is clicked so failure trajectories are visible
    local hasFailureTrajectories = false
    -- Check if there are existing failure trajectories
    local obbyFolder = getFolderElseCreate(Workspace, "ObbyParts" .. tostring(currentWorldIdx))
    local trajectoryFailureFolder = getFolderElseCreate(obbyFolder, "TrajectoryPartsFailure")
    if #trajectoryFailureFolder:GetChildren() > 0 then
        hasFailureTrajectories = true
    end
    
    -- Enable failure trajectory mode so failure trajectories will be visible when drawn
    inFailureTrajectoryMode[1] = true
    if showFailureTrajectoriesCheckbox then
        showFailureTrajectoriesCheckbox["checkmark"].Visible = true
    end

    local payload = prepareLevelParamsPayload()

    payload.max_traj_steps = maxTrajectorySteps
    payload.max_traj_jumps = maxTrajectoryJumps
    payload.traj_start_step = searchStartTimestep
    local bounds = CollectionService:GetTagged("SearchAABB")
    if #bounds > 0 then
        local boundsPart = bounds[1]
        payload.boundsPosition = {boundsPart.Position.X, boundsPart.Position.Y, boundsPart.Position.Z}
        payload.boundsExtents = {boundsPart.Size.X, boundsPart.Size.Y, boundsPart.Size.Z}
    end

    -- If the user selected a checkpoint, send its slot index and start search from there.
    local selected = SelectionService:Get()
    if #selected == 1 then
        local slot_idx = selected[1]:GetAttribute("slot_idx")
        if slot_idx ~= nil then
            payload.starting_slot_idx = slot_idx
        end
    end

    local jsonData = HttpService:JSONEncode(payload)

    -- Reset the polling flag
    shouldStopPolling[1] = false
    
    local pollInterval = 0.5 -- Poll every 0.5 seconds

    --Delete all existing bin visualizations
    local binVizBlocks = CollectionService:GetTagged("GoExploreViz")
    for _, b in binVizBlocks do
        b:Destroy()
    end
    bin_viz = {}
    goal_viz = {}
    intermediate_goal_viz = {}
    ckpt_viz = {}

    -- Helper function to process trajectory response
    local function processTrajectoryResponse(response)
        if not response then
            return false
        end
        local result = HttpService:JSONDecode(response)
        if result.total_trials == 0 then
            print("No trials")
            return false
        end
        local total = tick()
        
        local t = tick()
        -- Delete all existing trajectories.
        local obbyFolder = getFolderElseCreate(Workspace, "ObbyParts" .. tostring(currentWorldIdx))
        for _, trajectoryType in {"Success", "Human", "Failure"} do
            local currTrajectoryFolder = getFolderElseCreate(obbyFolder, "TrajectoryParts" .. trajectoryType)
            clearFolderContents(currTrajectoryFolder)
        end
        print("Profile: Delete Existing Trajectories "..tostring(tick() - t).." seconds")

        local function createCheckpointMarker(position, timestep, slot_idx)
            local repPoint = Instance.new("Part")
            repPoint.Size = Vector3.new(0.3,0.3,0.3)
            repPoint.Position = position
            repPoint.Color = Color3.fromRGB(150, 103, 102)
            repPoint.Transparency = 0.8
            repPoint.CanCollide = false
            repPoint.Anchored = true
            repPoint.CastShadow = false
            repPoint.Material = Enum.Material.Neon
            -- Name corresponds to representative timestep for this bin.
            repPoint.Name = tostring(timestep)
            repPoint:SetAttribute("slot_idx", slot_idx)
            return repPoint
        end
        local function createPositionHistoryMarker(position, idx, block)
            local p = Instance.new("Part")
            p.Size = Vector3.new(0.2,0.2,0.2)
            p.Position = position
            p.Color = Color3.fromRGB(255, 255, 0)
            p.Transparency = 0
            p.CanCollide = false
            p.Parent = block
            p.Anchored = true
            p.Name = "pos" .. tostring(idx)
            return p
        end

        t = tick()
        print("Profiler: JSON Decode "..tostring(tick() - t).." seconds")

        t = tick()
        if result.bin_size then
            local folder = Workspace:FindFirstChild("GoExploreBins")
            for idx, point in pairs(result.bin_points) do
                local position = Vector3.new(point[1], point[2], point[3])
                if not folder then
                    folder = Instance.new("Folder")
                    folder.Name = "GoExploreBins"
                    folder.Parent = Workspace
                end

            end

            for _, ckpt_entry in pairs(ckpt_viz) do
                ckpt_entry[1]:Destroy()
                ckpt_entry[2] = nil
            end
            -- Create the saved checkpoint markers.
            for idx, point in pairs(result.ckpt_points) do
                -- Create the point, associate the trajectory with that point.
                local position = Vector3.new(point[1], point[2], point[3])
                local repPoint = createCheckpointMarker(
                    position, point[4], result.slot_idx[idx]
                )
                CollectionService:AddTag(repPoint, "GoExploreViz")
                repPoint.Parent = showGoExploreBins[1] and folder or ServerStorage
                ckpt_viz[position] = {repPoint}
            end

            local function update_pbge_goal_viz(rep_points, viz_table, color)
                -- Set transparency to focus only on the currently active goals.
                for point, shape in pairs(viz_table) do
                    shape.Transparency = 1.0
                end
                for idx, point in pairs(rep_points) do
                    local goal_pos = Vector3.new(point[1], point[2], point[3])
                    -- Ensure no slight floating point mismatch.
                    local rounded = Vector3.new(math.ceil(goal_pos.X), math.ceil(goal_pos.Y), math.ceil(goal_pos.Z))
                    if viz_table[rounded] == nil then
                        local sphere = Instance.new("Part")
                        sphere.Size = Vector3.new(1,1,1)
                        sphere.Shape = Enum.PartType.Ball
                        sphere.Position = goal_pos
                        sphere.Color = color --Color3.fromRGB(0, 255, 0)
                        sphere.Transparency = 0.5
                        sphere.CollisionGroup = "GoExploreViz"
                        sphere.CanCollide = false
                        sphere.Parent = showGoExploreBins[1] and folder or ServerStorage
                        sphere.Anchored = true
                        CollectionService:AddTag(sphere, "GoExploreViz")
                        viz_table[rounded] = sphere
                    else
                        viz_table[rounded].Transparency = 0.5
                    end
                end
            end
            -- Update visualization for intermediate and final goals.
            update_pbge_goal_viz(result.goal_rep_points, goal_viz, Color3.fromRGB(0, 255, 0))
            update_pbge_goal_viz(result.intermediate_goal_rep_points, intermediate_goal_viz, Color3.fromRGB(255, 255, 0))
        end

        print("Profiler: Bin metadata "..tostring(tick() - t).." seconds")

        t = tick()
        -- Automatically enable trajectory modes if trajectories exist, so they're visible when drawn
        local hasSuccessTrajectories = result.success_trace_viz and #result.success_trace_viz > 0
        local hasFailureTrajectories = result.failure_trace_viz and #result.failure_trace_viz > 0
        
        if hasSuccessTrajectories or hasFailureTrajectories then
            -- When "Find Solution" is clicked, enable failure trajectory mode so they're visible
            -- Success trajectories take priority if both exist, but failure should also be visible
            if hasSuccessTrajectories then
                inSuccessTrajectoryMode[1] = true
                -- Also enable failure mode if failure trajectories exist
                if hasFailureTrajectories then
                    inFailureTrajectoryMode[1] = true
                else
                    inFailureTrajectoryMode[1] = false
                end
            else
                inSuccessTrajectoryMode[1] = false
                inFailureTrajectoryMode[1] = true
            end
            
            -- Update checkbox visuals if they exist
            if showSuccessfulTrajectoriesCheckbox then
                showSuccessfulTrajectoriesCheckbox["checkmark"].Visible = inSuccessTrajectoryMode[1]
            end
            if showFailureTrajectoriesCheckbox then
                showFailureTrajectoriesCheckbox["checkmark"].Visible = inFailureTrajectoryMode[1]
            end
        end
        print("Profiler: Update Success/Failure"..tostring(tick() - t).." seconds")

        t = tick()
        drawTrajectories(result, searchStartTimestep)
        print("Profiler: Draw Trajectories "..tostring(tick() - t).." seconds")

        print("Profiler: Total within processTrajectoryResponse "..tostring(tick()-total).." seconds")

        if hasSuccessTrajectories then
            searchReady()
        end
        
        return result.should_continue
    end

    -- Send findSuccessfulTrajectory request (this will block until it returns)
    local url = backendUrl .. "/findSuccessfulTrajectory"
    local success, response = pcall(HttpService.PostAsync, HttpService, url, jsonData)

    if success then
        while not shouldStopPolling[1] do
            -- Poll getCurrentTrajectory and process the response
            local pollUrl = backendUrl .. "/getCurrentTrajectory"
            local pollSuccess, pollResponse = pcall(HttpService.PostAsync, HttpService, pollUrl, jsonData)
            local shouldContinue = false
            if pollSuccess then
                local t = tick()
                shouldContinue = processTrajectoryResponse(pollResponse)
                print("Processed in: " .. tostring(tick() - t).." seconds")
            else
                print("Error polling current trajectory")
            end
            if shouldContinue then
                print("SHOULD KEEP GOING")
            end
            shouldStopPolling[1] = (not shouldContinue) or shouldStopPolling[1]
            -- Wait before next poll (check flag again after wait)
            wait(pollInterval)
        end
    end

    local result = {}
    if not success then
        print(response)
        print("Error finding successful trajectory")
        return
    end

    -- Get the final goExplore trajectory (one more time after findSuccessfulTrajectory completes)
    local url = backendUrl .. "/getCurrentTrajectory"
    local success, response = pcall(HttpService.PostAsync, HttpService, url, jsonData)
    if not success then
        print(response)
        print("Error fetching current trajectory")
        return
    end

    -- Process the final trajectory response
    processTrajectoryResponse(response)
    local result = HttpService:JSONDecode(response)

    simulationPaused[1] = false
    playOrPauseSimulation()

    local param = {
        param_name = "Num Trajectories",
        min_val = 0,
        curr_val = 8,
        max_val = result.total_trials,
        step = 1,
        enabled = true
    }

    numShownTrajectories = param.curr_val

    if numTrajectoriesSlider then
        numTrajectoriesSlider.destroy()
    end

    -- Automatically trigger Live Playback after receiving trajectories
    local hasSuccessTrajectories = result.success_trace_viz and #result.success_trace_viz > 0
    local hasFailureTrajectories = result.failure_trace_viz and #result.failure_trace_viz > 0
    
    if hasSuccessTrajectories or hasFailureTrajectories then
        -- Set trajectory mode based on what's available
        -- When "Find Solution" is clicked, keep failure trajectories visible
        if hasSuccessTrajectories then
            inSuccessTrajectoryMode[1] = true
            -- Enable live playback if a success has been found.
            inLivePlaybackMode[1] = true
            inFailureTrajectoryMode[1] = false
        else
            inSuccessTrajectoryMode[1] = false
            inFailureTrajectoryMode[1] = true
        end

        -- Update checkbox visuals
        if showSuccessfulTrajectoriesCheckbox then
            showSuccessfulTrajectoriesCheckbox["checkmark"].Visible = inSuccessTrajectoryMode[1]
        end
        if showFailureTrajectoriesCheckbox then
            showFailureTrajectoriesCheckbox["checkmark"].Visible = inFailureTrajectoryMode[1]
        end
        if livePlaybackCheckbox then
            livePlaybackCheckbox["checkmark"].Visible = inLivePlaybackMode[1]
        end
        
        -- Trigger the playback
        showSuccessOrFailureTrajectories()
    end

    return result
end

function showOrHideGoExploreBins()
    local folder = Workspace:FindFirstChild("GoExploreBins")
    for _, part in CollectionService:GetTagged("GoExploreViz") do
        part.Parent = showGoExploreBins[1] and folder or ServerStorage
        for _, p in pairs(part:GetChildren()) do
            -- Hide the previous representative points.
            if p.Transparency == 0.5 then
                p.Transparency = 1
            end
        end
    end
end

showSuccessfulTrajectoriesCheckbox = addCheckboxToFrame(UDim2.new(0, 80, 0.05, 20), "Success Traj.", inSuccessTrajectoryMode, showSuccessOrFailureTrajectories, analysisFrame)

local trajectoryIdxDropdownContainer = Instance.new("Frame")
trajectoryIdxDropdownContainer.Size = UDim2.new(0.07, 0, 0, 0) -- Will expand as we add options
trajectoryIdxDropdownContainer.Position = UDim2.new(0.9, 0, 0.1, 20)
trajectoryIdxDropdownContainer.BackgroundColor3 = Color3.fromRGB(240, 240, 240)
trajectoryIdxDropdownContainer.BorderSizePixel = 1
trajectoryIdxDropdownContainer.Visible = false
trajectoryIdxDropdownContainer.ZIndex = 6
trajectoryIdxDropdownContainer.Parent = analysisFrame
trajectoryIdxButton = createGUIButton("Idx", UDim2.new(0.07, 0, 0, 20), UDim2.new(0.9, 0, 0.1, 0), analysisFrame,
function()
    print("TRAJECTORY IDX BUTTON CLICKED")

    -- Clear existing options
    for _, child in pairs(trajectoryIdxDropdownContainer:GetChildren()) do
        child:Destroy()
    end

    local optionHeight = 20
    local obbyFolder = Workspace:FindFirstChild("ObbyParts" .. tostring(currentWorldIdx))
    local trajectorySuccessFolder = getFolderElseCreate(obbyFolder, "TrajectoryPartsSuccess")
    local offset = 0
    for i, c in ipairs(trajectorySuccessFolder:GetChildren()) do
        if not string.find(c.Name, tostring(i)) then
            continue
        end
        -- Used for placing the "All" button below.
        offset = i

        local option = Instance.new("TextButton")
        option.Text = tostring(i)
        option.Size = UDim2.new(1, 0, 0, optionHeight)
        option.Position = UDim2.new(0, 0, 0, (i - 1) * optionHeight)
        option.BackgroundColor3 = Color3.fromRGB(240, 240, 240)
        option.TextSize = 14
        option.Font = Enum.Font.SourceSans
        option.ZIndex = 6
        option.Parent = trajectoryIdxDropdownContainer

        option.MouseButton1Click:Connect(function()
            trajectoryIdxDropdownContainer.Visible = false
            activeTrajectoryIdx = i
            trajectoryIdxButton.Text = tostring(i)
            showSuccessOrFailureTrajectories()
        end)

        option.MouseEnter:Connect(function()
            option.BackgroundColor3 = Color3.fromRGB(220, 220, 220)
        end)

        option.MouseLeave:Connect(function()
            option.BackgroundColor3 = Color3.fromRGB(240, 240, 240)
        end)
    end

    -- Add a button to reset the index
    local option = Instance.new("TextButton")
    option.Text = "All"
    option.Size = UDim2.new(1, 0, 0, optionHeight)
    option.Position = UDim2.new(0, 0, 0, offset * optionHeight)
    option.BackgroundColor3 = Color3.fromRGB(240, 240, 240)
    option.TextSize = 14
    option.Font = Enum.Font.SourceSans
    option.ZIndex = 6
    option.Parent = trajectoryIdxDropdownContainer
    option.MouseButton1Click:Connect(function()
        trajectoryIdxDropdownContainer.Visible = false
        activeTrajectoryIdx = -1
        trajectoryIdxButton.Text = "All"
        showSuccessOrFailureTrajectories()
    end)
    option.MouseEnter:Connect(function()
        option.BackgroundColor3 = Color3.fromRGB(220, 220, 220)
    end)
    option.MouseLeave:Connect(function()
        option.BackgroundColor3 = Color3.fromRGB(240, 240, 240)
    end)
    trajectoryIdxDropdownContainer.Visible = not trajectoryIdxDropdownContainer.Visible
end)

showFailureTrajectoriesCheckbox = addCheckboxToFrame(UDim2.new(0.5, 40, 0.05, 20), "Failure Traj.", inFailureTrajectoryMode, showSuccessOrFailureTrajectories, analysisFrame)
livePlaybackCheckbox = addCheckboxToFrame(UDim2.new(0, 80, -0.1, 20), "Live Playback", inLivePlaybackMode, showSuccessOrFailureTrajectories, analysisFrame)
goExploreBinCheckbox = addCheckboxToFrame(UDim2.new(0.5, 40, -0.1, 20), "Show GoExplore Bins", buttonShowGoExploreBins, function()
    showGoExploreBins[1] = buttonShowGoExploreBins[1]
    showOrHideGoExploreBins()
end, analysisFrame)
pauseSimulationCheckbox = addCheckboxToFrame(UDim2.new(0.5, 40, 0.35, 20), "Pause Simulation", simulationPaused, playOrPauseSimulation, analysisFrame)

function createSearchAABB()
    local searchAABB = Instance.new("Part")
    searchAABB.Size = Vector3.new(10, 10, 10)
    searchAABB.CanCollide = false
    searchAABB.Anchored = true
    searchAABB.Color = Color3.fromRGB(255,255,255)
    searchAABB.Transparency = 0.9
    searchAABB.Parent = Workspace
    searchAABB.CastShadow = false
    CollectionService:AddTag(searchAABB, "SearchAABB")
    return searchAABB
end 
local addAABBForSearchButton = createGUIButton("Add Search Bound", UDim2.new(0.333333, -20, 0, 30), UDim2.new(0.666667, 0, 0, 130), contentFrame,
function()
    createSearchAABB()
end)

local loadQueryDropdownContainer = Instance.new("Frame")
loadQueryDropdownContainer.Size = UDim2.new(1, -40, 0, 0) -- Will expand as we add options
-- Place the list 5 pixels below the Load Query button (242 + 30).
loadQueryDropdownContainer.Position = UDim2.new(0, 20, 0, 277)
loadQueryDropdownContainer.BackgroundColor3 = Color3.fromRGB(240, 240, 240)
loadQueryDropdownContainer.BorderSizePixel = 1
loadQueryDropdownContainer.Visible = false
loadQueryDropdownContainer.ZIndex = 7
loadQueryDropdownContainer.Parent = contentFrame
local loadQueryButton = createGUIButton("Load Query", UDim2.new(1, -40, 0, 30), UDim2.new(0, 20, 0, 242), contentFrame,
function()
    if loadQueryDropdownContainer.Visible then
        loadQueryDropdownContainer.Visible = false
        return
    end
    loadQueryOptions = fetchLoadQueryOptions()
    if loadQueryOptions then
        updateLoadQueryOptions()
    end
    loadQueryDropdownContainer.Visible = not loadQueryDropdownContainer.Visible
end)

local findSolution = createGUIButton("Search", UDim2.new(0.5, -25, 0, 30), UDim2.new(0, 20, 0, 282), contentFrame,
function()
    -- Stop live playback when find solution is clicked
    stopLivePlayback()
    requestSolutionTrajectory()
end)
findSolution.BackgroundColor3 = Color3.fromRGB(50, 180, 50)

function searchReady()
    shouldStopPolling[1] = true
    findSolution.BackgroundColor3 = Color3.fromRGB(50, 180, 50)
    cancelRequestButton.BackgroundColor3 = Color3.fromRGB(100, 100, 100)
end

function searchNotReady()
    findSolution.BackgroundColor3 = Color3.fromRGB(100, 100, 100)
    cancelRequestButton.BackgroundColor3 = Color3.fromRGB(220, 60, 60)
end

-- When findSolution is clicked, make the button gray and cancelRequest button red.
findSolution.MouseButton1Click:Connect(searchNotReady)

cancelRequestButton.MouseButton1Click:Connect(function()
    searchReady()
end)

function updateBackendAgent(agentName)
    local agentValueToSend = agentName == "None" and "" or agentName
    local payload = HttpService:JSONEncode({
        agent = agentValueToSend
    })
    -- Update local display name
    currAgentName = agentValueToSend or ""
    print("currAgentName:", currAgentName)
    success, response = pcall(HttpService.PostAsync, HttpService, backendUrl .. "/updateBackendAgent", payload)
    print("agent updated successfully:", success)
end

function updateLoadQueryOptions()
    print("Updating load query options!!!")
-- Clear existing options
    for _, child in pairs(loadQueryDropdownContainer:GetChildren()) do
        child:Destroy()
    end

    -- Add new options
    local optionHeight = 25
    for i, optionText in ipairs(loadQueryOptions) do
        local option = Instance.new("TextButton")
        option.Text = string.sub(optionText, 1, string.find(optionText, ".json") - 1)
        option.Size = UDim2.new(1, 0, 0, optionHeight)
        option.Position = UDim2.new(0, 0, 0, (i - 1) * optionHeight)
        option.BackgroundColor3 = Color3.fromRGB(240, 240, 240)
        option.TextSize = 14
        option.Font = Enum.Font.SourceSans
        option.ZIndex = 6
        option.Parent = loadQueryDropdownContainer

        option.MouseButton1Click:Connect(function()
            -- Stop live playback when level is loaded
            stopLivePlayback()
            local success, response = pcall(HttpService.GetAsync, HttpService, backendUrl .. "/resetGoExplore")
            if not success then
                print("Error resetting go explore trajectory")
            end
            
            clearParameterSliders()
            loadQueryDropdownContainer.Visible = false
            
            -- Fetch the parameters and json for this level option
            local levelType, obstacleParams, levelJson, dslJson, max_traj_steps, max_traj_jumps, traj_start_step, bounds_position, bounds_size = requestSavedQueryParameters(optionText)

            -- To replay, the user should just have to press "Find Solution" once the query is loaded.
            maxTrajectorySteps = max_traj_steps
            updateTextBox(maxTrajStepsBox, tostring(maxTrajectorySteps))
            maxTrajectoryJumps = max_traj_jumps
            updateTextBox(maxTrajJumpsBox, tostring(maxTrajectoryJumps))
            -- Actual timestep will be updated below.
            searchStartTimestep = traj_start_step

            if bounds_position ~= nil then
                local searchAABB = createSearchAABB()
                searchAABB.Position = Vector3.new(bounds_position[1], bounds_position[2], bounds_position[3])
                searchAABB.Size = Vector3.new(bounds_size[1], bounds_size[2], bounds_size[3])
            end

            -- Update button state
            obstacleDropdown.Text = levelType
            currentLevelType[currentWorldIdx] = levelType
            currentParameters[currentWorldIdx] = obstacleParams
            -- The DSL json from which diffs will be computed.
            currentLevelJson[currentWorldIdx] = dslJson
            updateParameterSliders(currentParameters[currentWorldIdx])
            updateObstacleInstance(getFolderElseCreate(Workspace, "ObbyParts" .. tostring(currentWorldIdx)), levelJson, {})
            simulationPaused[1] = true
            playOrPauseSimulation()
            setSimulationTimestep(searchStartTimestep)
            wait(0.1) -- Wait to allow the sim to correctly set timesteps on objects.
        end)

        option.MouseEnter:Connect(function()
            option.BackgroundColor3 = Color3.fromRGB(220, 220, 220)
        end)

        option.MouseLeave:Connect(function()
            option.BackgroundColor3 = Color3.fromRGB(240, 240, 240)
        end)
    end
end

-- Function to update the dropdown with new level type options
function updateDropdownOptions()
    -- Clear existing options
    for _, child in pairs(obstacleDropdownContainer:GetChildren()) do
        child:Destroy()
    end

    local optionHeight = 25
    local yOffset = 0

    local function addHeader(text)
        local header = Instance.new("TextLabel")
        header.Text = text
        header.Size = UDim2.new(1, 0, 0, 25)
        header.Position = UDim2.new(0, 0, 0, yOffset)
        header.BackgroundTransparency = 0
        header.BackgroundColor3 = Color3.fromRGB(210, 210, 210)
        header.TextSize = 14
        header.Font = Enum.Font.SourceSansBold
        header.TextColor3 = Color3.fromRGB(35, 35, 35)
        header.TextXAlignment = Enum.TextXAlignment.Center
        header.TextYAlignment = Enum.TextYAlignment.Center
        header.ZIndex = 21
        header.Parent = obstacleDropdownContainer
        yOffset += 25
    end

    local function addOption(optionText, onClick)
        local option = Instance.new("TextButton")
        option.Text = optionText
        option.Size = UDim2.new(1, 0, 0, optionHeight)
        option.Position = UDim2.new(0, 0, 0, yOffset)
        option.BackgroundColor3 = Color3.fromRGB(240, 240, 240)
        option.TextSize = 14
        option.Font = Enum.Font.SourceSans
        option.ZIndex = 21
        option.Parent = obstacleDropdownContainer

        option.MouseButton1Click:Connect(function()
            onClick(optionText)
        end)

        option.MouseEnter:Connect(function()
            option.BackgroundColor3 = Color3.fromRGB(220, 220, 220)
        end)

        option.MouseLeave:Connect(function()
            option.BackgroundColor3 = Color3.fromRGB(240, 240, 240)
        end)

        yOffset += optionHeight
    end

        if loadLevelOptions and #loadLevelOptions > 0 then
        addHeader("Saved Levels")
        for _, optionText in ipairs(loadLevelOptions) do
            addOption(string.sub(optionText, 1, string.find(optionText, ".json") - 1), function(selectedSavedLevel)
                -- Stop live playback when level is loaded
                stopLivePlayback()
                local success, response = pcall(HttpService.GetAsync, HttpService, backendUrl .. "/resetGoExplore")
                if not success then
                    print("Error resetting go explore trajectory")
                end

                clearParameterSliders()
                obstacleDropdownContainer.Visible = false

                -- Fetch the parameters and json for this level option
                local levelType, obstacleParams, levelJson, dslJson = requestSavedLevelParameters(optionText)
                -- Update button state
                obstacleDropdown.Text = levelType
                currentLevelType[currentWorldIdx] = levelType
                currentParameters[currentWorldIdx] = obstacleParams
                -- The DSL json from which diffs will be computed.
                currentLevelJson[currentWorldIdx] = dslJson
                updateParameterSliders(currentParameters[currentWorldIdx])
                updateObstacleInstance(getFolderElseCreate(Workspace, "ObbyParts" .. tostring(currentWorldIdx)), levelJson, {})
            end)
        end
    end
    
    if levelTypeOptions and #levelTypeOptions > 0 then
        addHeader("Template Levels")
        for _, optionText in ipairs(levelTypeOptions) do
            addOption(optionText, function(selectedLevelType)
                -- Stop live playback when obstacle is changed
                stopLivePlayback()
                local success, response = pcall(HttpService.GetAsync, HttpService, backendUrl .. "/resetGoExplore")
                if not success then
                    print("Error resetting go explore trajectory")
                end

                clearParameterSliders()
                obstacleDropdown.Text = selectedLevelType
                obstacleDropdownContainer.Visible = false

                -- Fetch the parameters and json for this level option
                currentLevelType[currentWorldIdx] = selectedLevelType
                local obstacleParams, levelJson = requestLevelParameters()
                currentParameters[currentWorldIdx] = obstacleParams
                currentLevelJson[currentWorldIdx] = levelJson
                updateParameterSliders(currentParameters[currentWorldIdx])
                updateObstacleInstance(getFolderElseCreate(Workspace, "ObbyParts" .. tostring(currentWorldIdx)), currentLevelJson[currentWorldIdx], {})
            end)
        end
    end

    obstacleDropdownContainer.Size = UDim2.new(1, -40, 0, yOffset)

end

-- Function to update agent dropdown options
function updateAgentDropdownOptions()
    -- Clear existing options
    for _, child in pairs(agentDropdownContainer:GetChildren()) do
        child:Destroy()
    end

    local optionHeight = 25
    local yOffset = 0

    local function addOption(optionText, onClick)
        local option = Instance.new("TextButton")
        option.Text = optionText
        option.Size = UDim2.new(1, 0, 0, optionHeight)
        option.Position = UDim2.new(0, 0, 0, yOffset)
        option.BackgroundColor3 = Color3.fromRGB(240, 240, 240)
        option.TextSize = 14
        option.Font = Enum.Font.SourceSans
        option.ZIndex = 21
        option.Parent = agentDropdownContainer

        option.MouseButton1Click:Connect(function()
            onClick(optionText)
        end)

        option.MouseEnter:Connect(function()
            option.BackgroundColor3 = Color3.fromRGB(220, 220, 220)
        end)

        option.MouseLeave:Connect(function()
            option.BackgroundColor3 = Color3.fromRGB(240, 240, 240)
        end)

        yOffset += optionHeight
    end

    if agentOptions and #agentOptions > 0 then
        -- Add "None" option first
        addOption("None", function(selectedAgent)
            currAgentName = "None"
            agentDropdown.Text = "None"
            agentDropdownContainer.Visible = false
            print("Selected agent:", currAgentName)
            updateBackendAgent(selectedAgent)
        end)
        
        for _, optionText in ipairs(agentOptions) do
            addOption(optionText, function(selectedAgent)
                currAgentName = selectedAgent
                agentDropdown.Text = selectedAgent
                agentDropdownContainer.Visible = false
                print("Selected agent:", currAgentName)
                updateBackendAgent(selectedAgent)
            end)
        end
    end

    agentDropdownContainer.Size = UDim2.new(1, -40, 0, yOffset)

end
-- Function to create a slider for a parameter
-- Configurable update behavior with a custom update callback.
function createParameterSlider(parameter, yPos, container, customUpdateCallback)
    local leftOffset = 0.1

    -- Slider frame
    local sliderFrame = Instance.new("Frame")
    sliderFrame.Size = UDim2.new(0.85, 0, 0, 6)
    sliderFrame.Position = UDim2.new(leftOffset, 0, 0, yPos + 25)
    sliderFrame.BackgroundColor3 = Color3.fromRGB(200, 200, 200)
    sliderFrame.BorderSizePixel = 0
    sliderFrame.Parent = container

    -- Add checkbox to left of everything
    local checkbox = Instance.new("Frame")
    checkbox.Size = UDim2.new(0, 12, 0, 12)
    checkbox.Position = UDim2.new(0.025, 0.0, 0.0, yPos + 10)
    checkbox.BackgroundColor3 = Color3.fromRGB(255, 255, 255)
    checkbox.BorderSizePixel = 1
    checkbox.BorderColor3 = Color3.fromRGB(27, 42, 53)
    checkbox.Visible = true 
    checkbox.Parent = container
    -- Actual checkmark
    local checkboxCheckmark = Instance.new("TextLabel")
    checkboxCheckmark.Size = UDim2.new(1, 0, 1, 0)
    checkboxCheckmark.BackgroundTransparency = 1
    checkboxCheckmark.Text = "✓"
    checkboxCheckmark.TextSize = 18
    checkboxCheckmark.TextColor3 = Color3.fromRGB(0, 0, 0)
    checkboxCheckmark.Font = Enum.Font.SourceSansBold
    checkboxCheckmark.Visible = parameter.enabled
    checkboxCheckmark.Parent = checkbox
    -- Create the button to make it clickable
    local checkboxButton = Instance.new("TextButton")
    checkboxButton.Size = UDim2.new(1, 0, 1, 0)
    checkboxButton.BackgroundTransparency = 1
    checkboxButton.Text = ""
    checkboxButton.Parent = checkbox
    -- when clicked, make enabled/disabled
    checkboxButton.MouseButton1Click:Connect(function()
        checkboxCheckmark.Visible = not checkboxCheckmark.Visible
        parameter.enabled = not parameter.enabled
    end)

    -- Slider knob
    local knob = Instance.new("Frame")
    knob.Size = UDim2.new(0, 8, 2.5, 0)
    local range = parameter.max_val - parameter.min_val
    local initialPos = (parameter.curr_val - parameter.min_val) / range
    knob.Position = UDim2.new(initialPos, -4, 0, -5)
    knob.BackgroundColor3 = Color3.fromRGB(50, 120, 220)
    knob.BorderSizePixel = 0
    knob.Parent = sliderFrame
    -- Circular
    local cornerRadius = Instance.new("UICorner")
    cornerRadius.CornerRadius = UDim.new(1, 0)
    cornerRadius.Parent = knob

    -- Use this to create labels
    local function createLabel(text, size, position, fontsize, font, alignment, parent)
        local label = Instance.new("TextLabel")
        label.Text = text
        label.Size = size
        label.Position = position
        label.BackgroundTransparency = 1
        label.TextSize = fontsize
        label.Font = font
        label.TextXAlignment = alignment
        label.Parent = parent
        return label
    end
    local nameLabel = createLabel(parameter.param_name, UDim2.new(0.95, 0, 0, 20), UDim2.new(leftOffset, 0, 0, yPos), 14,
        Enum.Font.SourceSansBold, Enum.TextXAlignment.Left, container)
    local valueLabel = createLabel(string.format("%.2f", tostring(parameter.curr_val)), UDim2.new(0.2, 0, 0, 20),
        UDim2.new(0.78, 0, 0, yPos), 14, Enum.Font.SourceSans, Enum.TextXAlignment.Right, container)
    local minLabel = createLabel(string.format("%.2f", tostring(parameter.min_val)), UDim2.new(0.1, 0, 0, 20),
        UDim2.new(leftOffset, 0, 0, yPos + 30), 12, Enum.Font.SourceSans, Enum.TextXAlignment.Left, container)
    local maxLabel = createLabel(string.format("%.2f", tostring(parameter.max_val)), UDim2.new(0.1, 0, 0, 20),
        UDim2.new(0.875, 0, 0, yPos + 30), 12, Enum.Font.SourceSans, Enum.TextXAlignment.Right, container)

    -- Draw some ticks
    local num_ticks = math.floor(range / parameter.step)
    for i = 0, num_ticks do
        local tick = Instance.new("Frame")
        tick.Size = UDim2.new(0, 1, 0, 6)
        tick.Position = UDim2.new(i / num_ticks, -0.5, 0, -5)
        tick.BackgroundColor3 = Color3.fromRGB(100, 100, 100)
        tick.BorderSizePixel = 0
        tick.Parent = sliderFrame
    end

    -- Value helper only performs clamping.
    local function updateValueHelper(value)
        value = math.clamp(value, parameter.min_val, parameter.max_val)

        -- Format for display
        local displayValue
        if parameter.step < 0.1 then
            displayValue = string.format("%.2f", value)
        else
            displayValue = string.format("%.1f", value)
        end
        -- Update text and knob
        valueLabel.Text = displayValue
        local newPos = (value - parameter.min_val) / (parameter.max_val - parameter.min_val)
        knob.Position = UDim2.new(newPos - 0.025, 0, 0, -5)

        -- Store value
        parameter.curr_val = value

        if customUpdateCallback then
            customUpdateCallback()
        end
    end

    -- Update fun when the slider is moved
    local function convertMousePositionAndUpdateValue(xPos)
        local sliderWidth = sliderFrame.AbsoluteSize.X
        local relativeX = math.clamp(xPos - sliderFrame.AbsolutePosition.X, 0, sliderWidth)
        local percent = relativeX / sliderWidth

        -- Calculate absolute value
        local value = parameter.min_val + (parameter.max_val - parameter.min_val) * percent

        -- Round to step
        if parameter.step > 0 then
            value = math.floor(value / parameter.step + 0.5) * parameter.step
        end

        updateValueHelper(value)
    end

    sliderFrame.InputBegan:Connect(function(input)
        if input.UserInputType == Enum.UserInputType.MouseButton1 then
            convertMousePositionAndUpdateValue(input.Position.X)
        end
    end)

    return {
        height = 60, -- Height of this parameter's UI elements
        sliderFrame = sliderFrame,
        update = function(value)
            updateValueHelper(value)
        end,
        destroy = function()
            nameLabel:Destroy()
            valueLabel:Destroy()
            sliderFrame:Destroy()
            minLabel:Destroy()
            maxLabel:Destroy()
        end
    }
end

function clearParameterSliders()
    for _, child in pairs(parametersScrollFrame:GetChildren()) do
        child:Destroy()
    end
    parametersScrollFrame.Visible = false
    analysisFrame.Visible = false
    blockEditorFrame.Visible = false
end

-- Update the UI sliders given the parameters [parameters]
function updateParameterSliders(parameters)
    -- Guard against the first call before anything has been loaded.
    if not parameters then
        return
    end
    analysisFrame.Visible = true
    blockEditorFrame.Visible = true

    -- Create sliders for each parameter
    local i = 1
    local totalHeight = 0
    for _, param in parameters do
        if param.distribution ~= "Constant" then
            -- Make the slider UI (and store)
            local yPos = (i - 1) * 60
            local slider = createParameterSlider(param, yPos, parametersScrollFrame, function()
                -- Update the level geometry
                currentLevelJson[currentWorldIdx] = sendParametersAndUpdateLevel()
            end)
            param.slider = slider
            totalHeight = totalHeight + slider.height
            i = i + 1
        end
    end

    -- Update scrolling frame canvas size
    parametersScrollFrame.CanvasSize = UDim2.new(0, 0, 0, totalHeight)
    roundCorners(parametersScrollFrame, 0.03)
end

-- Button click handler
button.Click:Connect(function()
    widget.Enabled = not widget.Enabled
    if widget.Enabled then
        levelTypeOptions = fetchLevelOptions()
        agentOptions = fetchAgentOptions()
        if levelTypeOptions and agentOptions then
            updateDropdownOptions()
        end
    end
end)

updateDropdownOptions()

-- Ensure that character's transparency gets updated correctly when it spawns.
local player = PlayerCloneModule.getCurrentPlayer()
player.CharacterAdded:Connect(function(character)
    updatePlayerAndDynamicObject()
end)


