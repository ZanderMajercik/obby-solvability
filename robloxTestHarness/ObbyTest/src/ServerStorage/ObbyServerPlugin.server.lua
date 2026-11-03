assert(plugin, "This script must be run as a plugin!")
if not plugin then
	return
end

local ChangeHistoryService = game:GetService("ChangeHistoryService")
local Selection = game:GetService("Selection")
local CollectionService = game:GetService("CollectionService")
local httpSrv = game:GetService("HttpService")
local ServerStorage = game:GetService("ServerStorage")
local HTTPSettings = require(ServerStorage.HTTPSettings)

local LoadObbyFunctions = require(ServerStorage.LoadObbyFunctions)
local TrainingBehaviorFunctions = require(ServerStorage.TrainingBehaviorFunctions)


local episode = {
	success = false,
	observations = {},
	actions = {}
}

local MAX_TAGGED_SECTIONS = 100

local function decodeWrapper(stringtodecode)
	return httpSrv:JSONDecode(stringtodecode)
end

-- This code can be pasted into the command bar, but only once
local pluginMenu = plugin:CreatePluginMenu(math.random(), "Test Menu")
pluginMenu.Name = "Test Menu"

pluginMenu:AddNewAction("Size", "Size", "")
pluginMenu:AddNewAction("Orientation", "Orientation", "")
pluginMenu:AddNewAction("Position", "Position", "")

local properties = {"Size", "Position", "Orientation"}
local propertyChangedConnection = nil
local selectionChangedConnection = nil
local active = false

local transforms = {
	Vector3.new(1,0,0),
	Vector3.new(-1,0,0),
	Vector3.new(0,1,0),
	Vector3.new(0,-1,0),
	Vector3.new(0,0,1),
	Vector3.new(0,0,-1)
}

local function modifyProperty(p_i, property, i)
	if property == "Size" then
		p_i[property] += 5 * Vector3.new(math.abs(transforms[i].X), math.abs(transforms[i].Y), math.abs(transforms[i].Z))
		if property == "Size" then
			-- Match the tool behavior in Roblox
			p_i.Position += 0.5 * (5 * (p_i.CFrame.Rotation * transforms[i]))
		end
	end
	if property == "Orientation" then
		p_i[property] += 90 * transforms[i]
	end
	if property == "Position" then
		p_i[property] += 5 * transforms[i]
	end
end

local visualizationObjects = {}
-- difficultyDeltas is a vector of 6 difficulty rating deltas relative to the difficulty rating of the non-transformed block.
-- They are directly used to color the block.
local function updateVisualization(part, property, difficultyDeltas)
	local currentPropertyValue = part[property]
	
	-- Each tool presents 6 possible snap transforms by default.
	for i = 1, 6, 1 do
		if visualizationObjects[i] then
			visualizationObjects[i]:Destroy()
		end
		local p_i = part:Clone()
		-- Remove all tags to prevent this object being sent.
		for _, tag in CollectionService:GetTags(p_i) do
			CollectionService:RemoveTag(p_i, tag)
		end
		modifyProperty(p_i, property, i)
		local difficultyDelta = difficultyDeltas[i]
		local blueMultiplier = 1.0
		-- Visualization tweaks.
		if difficultyDelta > 0.2 then
			difficultyDelta *= 3
			blueMultiplier = 0.5
		end
		p_i.Transparency = 0.5
		p_i.Parent = workspace
		p_i.Color = Color3.new(math.clamp(p_i.Color.R + difficultyDelta, 0.0, 1.0), math.clamp(p_i.Color.G - difficultyDelta, 0.0, 1.0), p_i.Color.B * blueMultiplier)
		p_i.Locked = true -- Prevent the parts from being selected in studio.
		p_i.CanCollide = false -- Prevent the parts from exporting with the level geometry.
		p_i.Size = p_i.Size * 1.01 -- Prevent Z-fighting.
		visualizationObjects[i] = p_i
	end
	
end

local function measureDifficultyMultisample(property)
	print("Measuring difficulty...")

	-- It should be possible to select multiple objects without a problem on
	-- the backend though the visualization doesn't support this currently.
	if #Selection:Get() ~= 1 then
		print("Must select exactly one object.")
		return
	end

	local highestSectionTagValue = -1
	local current_selection = nil
	for _, part in Selection:Get() do
		current_selection = part
		local partTags = CollectionService:GetTags(part)
		for _, tag in partTags do
			local sectionNumber = tonumber(tag)
			if sectionNumber then
				highestSectionTagValue = math.max(highestSectionTagValue, sectionNumber)
			end
		end
	end

	if highestSectionTagValue == -1 then
		print("No selection, or failed to find section tag.")
		return
	end
	
	-- Get the successful trajectory
	local obbyDescription = LoadObbyFunctions.createJsonFromObbySection(highestSectionTagValue)

	for partName in obbyDescription do
		partDesc = obbyDescription[partName]
		if current_selection.Position.X == partDesc["position"][1] and 
		   current_selection.Position.Y == partDesc["position"][2] and 
		   current_selection.Position.Z == partDesc["position"][3] then
			table.insert(obbyDescription[partName]["behaviors"], {
				conditions={},
				name="Transform" .. property
			})
		end
	end

	print(obbyDescription)

	local obbyDescriptionAndLabel = {
		name = "Section" .. tostring(highestSectionTagValue),
		obbyJson = obbyDescription,
		targetPosition = {}
	}

	local t = tick()
	-- Response should be the difficulty
	local success, response = pcall(httpSrv.PostAsync, httpSrv,HTTPSettings.baseURLAndPort .. "measureDifficultyMultisample", httpSrv:JSONEncode(obbyDescriptionAndLabel))
	if success then
		print("RBX Time: ", tick() - t)
		print(response)
	else
		print("HTTP Request failed.")
	end
	print(type(response))
	local successRates = decodeWrapper(string.sub(response, string.find(response, "%["), #response))
	return successRates
end

local function measureDifficulty()
	print("Measuring difficulty...")

	local highestSectionTagValue = -1
	for _, part in Selection:Get() do
		local partTags = CollectionService:GetTags(part)
		for _, tag in partTags do
			local sectionNumber = tonumber(tag)
			if sectionNumber then
				highestSectionTagValue = math.max(highestSectionTagValue, sectionNumber)
			end
		end
	end

	if highestSectionTagValue == -1 then
		print("No selection, or failed to find section tag.")
		return
	end

	-- Get the successful trajectory
	local obbyDescription = LoadObbyFunctions.createJsonFromObbySection(highestSectionTagValue)
	local obbyDescriptionAndLabel = {
		name = "Section" .. tostring(highestSectionTagValue),
		obbyJson = obbyDescription,
		targetPosition = {}
	}

	local t = tick()
	-- Response should be the difficulty
	local success, response = pcall(httpSrv.PostAsync, httpSrv,HTTPSettings.baseURLAndPort .. "measureDifficulty", httpSrv:JSONEncode(obbyDescriptionAndLabel))
	if success then
		print("RBX Time: ", tick() - t)
		print(response)
	else
		print("HTTP Request failed.")
	end
end

local function connectPropertyChanged(part, property)
	return part:GetPropertyChangedSignal("CFrame"):Connect(function()
		print("Changed ", property, ":", part.CFrame)
		local successRatings = measureDifficultyMultisample(property)
		local normalizedDifficultyDiffs = {}
		local eps = 0.0001
		for i = 2, 7, 1 do
			-- Iterate through the 6 transforms, computing the normalized delta.
			normalizedDifficultyDiffs[i - 1] = (successRatings[1] - successRatings[i]) / (successRatings[1] + eps)
		end
		 -- Will pass the result of this to the visualization for update.
		updateVisualization(part, property, normalizedDifficultyDiffs)
	end)
end

local function difficultyGradient()
	if active then
		active = false
		-- Remove connections and message saying service is deactivated
		if selectionChangedConnection then
			selectionChangedConnection:Disconnect()
			selectionChangedConnection = nil
		end
		if propertyChangedConnection then
			propertyChangedConnection:Disconnect()
			propertyChangedConnection = nil
		end

		-- Remove all visualizations.
		for i = 1, 6, 1 do
			if visualizationObjects[i] then
				visualizationObjects[i]:Destroy()
			end
		end

		print("Difficulty Gradient OFF")
		return
	end

	active = true

	local selectedAction = pluginMenu:ShowAsync()
	if not selectedAction then
		print("User did not select an action!")
		active = false
		return
	end

	-- If a single object is selected, track its position.
	if #Selection:Get() == 1 then
		if propertyChangedConnection then
			propertyChangedConnection:Disconnect()
			propertyChangedConnection = nil
		end
	
		for _, part in Selection:Get() do
			propertyChangedConnection = connectPropertyChanged(part, selectedAction.Text)
		end
	end
		
	-- Ensure correct switching when selection changed. 
	selectionChangedConnection = Selection.selectionChanged:Connect(function()
		if #Selection:Get() ~= 1 then
			print("Must select exactly one object.")
			return
		end

		if propertyChangedConnection then
			propertyChangedConnection:Disconnect()
			propertyChangedConnection = nil
		end

		for _, part in Selection:Get() do
			propertyChangedConnection = connectPropertyChanged(part, selectedAction.Text)
		end
	end)

	print("Difficulty Gradient ON")
end

--Change to trigger recompile.
local function getNewObby(serverSetupData, setGoalLogic)

	if not setGoalLogic then
		setGoalLogic = true
	end

	for idx, p in CollectionService:GetTagged("obby") do
		p:ClearAllChildren()
		p:Destroy()
	end

	--Load a json-style level from the server
	local success, response = pcall(httpSrv.GetAsync, httpSrv, HTTPSettings.baseURLAndPort .. "sendJsonDescription")
	if success then
		local b, data =  pcall(decodeWrapper, response)
	
		levelAABBs = {}
		-- Load the required obby.
		LoadObbyFunctions.createObbyFromJson(data, levelAABBs)
		print(levelAABBs)
	
		if setGoalLogic then 
			-- Set up goal logic.
			local lastCheckpoint = nil
			for idx, p in CollectionService:GetTagged("Goal") do
				if p.parent ~= workspace then
					continue
				end
				TrainingBehaviorFunctions["KillAndReportEpisode"](p, episode)
			end
		end
	end
    return success
end


-- See: https://create.roblox.com/docs/reference/engine/datatypes/CFrame#fromMatrix
local convertToZUpFrame = CFrame.fromMatrix(
    Vector3.new(0, 0, 0),
    Vector3.new(1, 0, 0), -- +Right Vector
    Vector3.new(0, 0, 1), -- +Up Vector
    -Vector3.new(0, 1, 0) -- -Forward Vector
)

local function convertToYUp(v)
	return convertToZUpFrame:Inverse() * v
end

local function exportLevelSections()
	-- Export level functionality. Assume there will be no more than 100 tagged sections.
    for i = 1, MAX_TAGGED_SECTIONS, 1 do
        local obbyDescription = LoadObbyFunctions.createJsonFromObbySection(i)
		if next(obbyDescription) == nil then
			return
		end
        local obbyDescriptionAndLabel = {
            name = "Section" .. tostring(i),
			folder = "RobloxExports",
            obbyJson = obbyDescription
        }
        print(obbyDescription)
        httpSrv:PostAsync(HTTPSettings.baseURLAndPort .. "receiveJsonDescription", httpSrv:JSONEncode(obbyDescriptionAndLabel))
    end
end


local function sendDescriptionAndDrawTrajectory(obbyDescriptionAndLabel)
	-- If successful, response should contain a list of positions off of which we can get parts.
	local success, response = pcall(httpSrv.PostAsync, httpSrv,HTTPSettings.baseURLAndPort .. "displaySuccessfulTrajectory", httpSrv:JSONEncode(obbyDescriptionAndLabel))
	if success then
		if response == "FAIL" then
			print("Failed to find a successful trajectory!")
		end
		local trajectoryPositions = httpSrv:JSONDecode(response)
		print(trajectoryPositions)

		-- Try to begin a recording with a specific identifier
		local recording = ChangeHistoryService:TryBeginRecording("Collect level for solution export")
		local trajectoryFolder = Instance.new("Folder", workspace)
		trajectoryFolder.Name = "Trajectory" .. obbyDescriptionAndLabel["name"]
		-- Add new part.
		for _, pos in trajectoryPositions do
			local part = Instance.new("Part")
			part.Color = Color3.new(1,1,0)
			part.Size = Vector3.new(0.5,0.5,0.5)
			part.Position = convertToYUp(Vector3.new(pos[1], pos[2], pos[3]))
			--part.Position = convertToYUp(Vector3.new(t.position[1], t.position[2], t.position[3]))
			part.Parent = trajectoryFolder
			part.Anchored = true
			part.CollisionGroup = "TrajectoryViz"
			--part.Transparency = 0.5
		end
		-- Finish the recording, committing the changes to the history
		ChangeHistoryService:FinishRecording(recording, Enum.FinishRecordingOperation.Commit)
	else
		print("HTTP Request failed.")
	end
end

local function generateSolutionTrajectory()
	print("STARTING SOLVE...")

	local highestSectionTagValue = -1
	for _, part in Selection:Get() do
		local partTags = CollectionService:GetTags(part)
		for _, tag in partTags do
			local sectionNumber = tonumber(tag)
			if sectionNumber then
				highestSectionTagValue = math.max(highestSectionTagValue, sectionNumber)
			end
		end
	end

	if highestSectionTagValue == -1 then
		print("No selection, or failed to find section tag.")
		return
	end

	-- Clears all in progress recordings. Shouldn't be necessary because the only thing recorded is creating the trajectory points.
	--ChangeHistoryService:SetEnabled(false)
	--ChangeHistoryService:SetEnabled(true)
	-- if ChangeHistoryService:IsRecordingInProgress("Collect level for solution export") then
	-- 	print("Recording was in progress!")
	-- 	-- Finish the recording, committing the changes to the history
	-- 	ChangeHistoryService:FinishRecording(recording, Enum.FinishRecordingOperation.Cancel)
	-- end


	-- Check if recording was successfully initiated
	-- if not recording then
	-- 	print("Failed to start recording")
	-- 	return
	-- 	-- ChangeHistoryService:FinishRecording("Collect level for solution export", Enum.FinishRecordingOperation.Cancel)
	-- 	-- recording = ChangeHistoryService:TryBeginRecording("Collect level for solution export")
	-- end

	-- Get the successful trajectory
	local obbyDescription = LoadObbyFunctions.createJsonFromObbySection(highestSectionTagValue)
	local obbyDescriptionAndLabel = {
		name = "Section" .. tostring(highestSectionTagValue),
		obbyJson = obbyDescription,
		targetPosition = {}
	}

	sendDescriptionAndDrawTrajectory(obbyDescriptionAndLabel)
end


local function generateSolutionTrajectoryThroughTarget()

	if #Selection:Get() ~= 1 then
		print("Must select exactly one object.")
		return
	end

	local targetPart = nil
	local highestSectionTagValue = -1
	for _, part in Selection:Get() do
		targetPart = part
		local partTags = CollectionService:GetTags(part)
		for _, tag in partTags do
			local sectionNumber = tonumber(tag)
			if sectionNumber then
				highestSectionTagValue = math.max(highestSectionTagValue, sectionNumber)
			end
		end
	end

	if highestSectionTagValue == -1 then
		print("No selection, or failed to find section tag.")
		return
	end

	-- Export the obby geometry.
	local obbyDescription = LoadObbyFunctions.createJsonFromObbySection(highestSectionTagValue)
	local obbyDescriptionAndLabel = {
		name = "Section" .. tostring(highestSectionTagValue),
		obbyJson = obbyDescription,
		targetPosition = {targetPart.Position.X, targetPart.Position.Y, targetPart.Position.Z}
	}
	print(obbyDescription)

	sendDescriptionAndDrawTrajectory(obbyDescriptionAndLabel)
end


-- PLUGIN CREATION
local toolbar = plugin:CreateToolbar("Demo")

local generateSolutionTrajectoryButton = toolbar:CreateButton("Solve Section", "", "")
local generateSolutionTrajectoryThroughTargetButton = toolbar:CreateButton("Solve Through Selected Block", "", "")
local measureDifficultyButton = toolbar:CreateButton("Measure Difficulty", "", "")
local difficultyGradientButton = toolbar:CreateButton("Difficulty Gradient", "", "")
local exportLevelSectionsButton = toolbar:CreateButton("Export Tagged Sections To JSON", "", "")

-- Connect function to buttons.
generateSolutionTrajectoryButton.Click:Connect(generateSolutionTrajectory)
generateSolutionTrajectoryThroughTargetButton.Click:Connect(generateSolutionTrajectoryThroughTarget)
measureDifficultyButton.Click:Connect(measureDifficulty)
difficultyGradientButton.Click:Connect(difficultyGradient)
exportLevelSectionsButton.Click:Connect(exportLevelSections)