local OrientationArrow = {}
local CollectionService = game:GetService("CollectionService")

function OrientationArrow.createOrientationArrow(self)
    -- Draw an arrow on top of the player's head
    local arrow = Instance.new("Part", workspace)
    arrow.Size = Vector3.new(0.3, 0.3, 1.5)
    arrow.Position = Vector3.new(0, 2.5, -1)
    arrow.Anchored = true
    arrow.BrickColor = BrickColor.new("Bright yellow")
    arrow.Material = Enum.Material.Metal
    arrow.CanCollide = false
    arrow.Parent = workspace
    arrow.Name = "OrientationArrow"

    -- Create a black tip for the arrow
    local tip = Instance.new("Part", arrow)
    tip.Size = Vector3.new(0.3, 0.3, 0.3)
    tip.Position = Vector3.new(0, 0, 0.6)
    tip.Anchored = true
    tip.BrickColor = BrickColor.new("Black")
    tip.Material = Enum.Material.Metal
    tip.CanCollide = false
    tip.Parent = arrow

    CollectionService:AddTag(arrow, "OrientationArrow")
    CollectionService:AddTag(tip, "OrientationArrowTip")
end

function OrientationArrow.destroyOrientationArrow(self)
    -- Destroy the current orientation arrow
    for i, c in CollectionService:GetTagged("OrientationArrow") do
        c:Destroy()
    end
    for i, c in CollectionService:GetTagged("OrientationArrowTip") do
        c:Destroy()
    end
end

function OrientationArrow.UpdateOrientation(self, updateCFrame, character)
	local cameraOrientation = updateCFrame
	local cameraDirection = cameraOrientation.LookVector
	-- Zero out vertical component
	cameraDirection = Vector3.new(cameraDirection.X, 0, cameraDirection.Z).Unit

	for i, arrow in CollectionService:GetTagged("OrientationArrow") do
		-- Draw an arrow on top of the player's head
		arrow.Position = character.HumanoidRootPart.Position + Vector3.new(cameraDirection.X, 2.5, cameraDirection.Z)
		arrow.CFrame = CFrame.new(arrow.Position, arrow.Position + cameraDirection)
    end
    for i, tip in CollectionService:GetTagged("OrientationArrowTip") do
        -- Draw an arrow on top of the player's head
        tip.Position = character.HumanoidRootPart.Position + Vector3.new(cameraDirection.X, 2.5, cameraDirection.Z) + Vector3.new(cameraDirection.X, 0, cameraDirection.Z)
        tip.CFrame = CFrame.new(tip.Position, tip.Position + cameraDirection)
    end
end

function OrientationArrow.GetCFrame(self)
    for i, arrow in CollectionService:GetTagged("OrientationArrow") do
        return arrow.CFrame
    end
end


return OrientationArrow